from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from transformers import (
    AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig,
    LlavaForConditionalGeneration, LlavaNextProcessor,GenerationConfig,
    LlavaNextForConditionalGeneration, Qwen2VLForConditionalGeneration,AutoModel, AutoTokenizer
)
from PIL import Image
import torch
import pandas as pd
import re
from collections import defaultdict
import numpy as np
from sklearn.metrics import accuracy_score
import ast
from PIL import Image
import matplotlib.pyplot as plt
import gc

from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

# JANUS IMPORTS
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
import base64

import argparse
import os
import pickle

from collections import defaultdict
import subprocess
from qwen_vl_utils import process_vision_info
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def get_nvidia_smi():
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    return result.stdout

def clean_janus_answer(text):
    pattern = r"<\|Assistant\|>:\s?(.*?)(?=<｜end▁of▁sentence｜>)"
    match = re.search(pattern, text)
    return match.group(1).strip('.')

def clean_instruction_tokens(text):
    cleaned_text = re.sub(r'\[INST\]\s*\n?.*?\[/INST\]\s*', '', text, flags=re.DOTALL)
    return cleaned_text.strip()

def get_image_token_indices(model_version, input_ids, tokenizer):
    if model_version == "llava-next":
        img_token_id = tokenizer.convert_tokens_to_ids("<image>")
        img_indices = np.where(input_ids == img_token_id)[0]
        text_indices = np.where(input_ids != img_token_id)[0]
        return img_indices, text_indices

    elif model_version == "qwen":
        vision_start = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        vision_end = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        vision_start_pos = np.where(input_ids == vision_start)[0]
        vision_end_pos = np.where(input_ids == vision_end)[0]
        if len(vision_start_pos) == 0 or len(vision_end_pos) == 0:
            raise ValueError("Could not find <|vision_start|> or <|vision_end|> tokens")
        img_indices = np.arange(vision_start_pos[0], vision_end_pos[0] + 1)
        text_indices = np.setdiff1d(np.arange(len(input_ids)), img_indices)
        return img_indices, text_indices

    elif model_version == "janus":
        img_token_id = tokenizer.convert_tokens_to_ids("<image_placeholder>")
        img_indices = np.where(input_ids == img_token_id)[0]
        text_indices = np.where(input_ids != img_token_id)[0]
        return img_indices, text_indices

    else:
        raise ValueError(f"Unknown model_version: {model_version}")

def run_attention_mass_analysis(batch_df, task, processor, model, model_version, image_type, most, instruction_tokens, end_tokens):
    from qwen_vl_utils import process_vision_info
    from collections import defaultdict
    import torch, numpy as np, base64, gc
    from PIL import Image

    results = {}

    for idx, row in batch_df.iterrows():
        entry = defaultdict(list)

        # --- Image path ---
        if task == "size":
            image_path = row['path_to_counterfact'] if image_type == 'counterfact' else row['path_to_clean']
        else:
            base_path = "/oscar/data/ceickhof/visual_counterfact/"
            if image_type == 'counterfact':
                color_to_replace = row['image_path'].split('_')[1]
                new_path = row['image_path'].replace(color_to_replace, row['incorrect_answer'])
                image_path = new_path.replace('downloaded_images', 'downloaded_images_counterfact')
                image_path = base_path + "final_counterfact_images/" + image_path
            else:
                image_path = row['image_path']

        # --- Load image ---
        try:
            if model_version == "janus":
                with open(image_path, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode("utf-8")
                    image = f"data:image/jpeg;base64,{image_data}"
            else:
                image = Image.open(image_path).convert("RGB")
                image = image.resize((256, 200) if task == "size" else (256, 256), Image.LANCZOS)
        except FileNotFoundError:
            print(f"Warning: Image not found at {image_path}")
            continue

        # --- Build prompt ---
        if task == "size":
            prompt = row["prompt_most"] if most == "True" else row["prompt_this"]
            prompt = f"{instruction_tokens} {prompt} {end_tokens}"
        else:
            """
            object_name = row['correct_object']
            question = f"What color is {'a' if most == 'True' else 'this'} {object_name}?"
            prompt = f"{instruction_tokens} Answer with one word. {question} {end_tokens}"
            """
            object_name = row['correct_object']
            #question = f"What color is {'a' if most == 'True' else 'this'} {object_name}?"
            if most == "True":
                #question = f"What color are most {object_name}s?"
                object_name_plural = object_name if object_name.endswith("s") else object_name + "s"
                question = f"What color are most {object_name_plural}?"
                
            else:
                question = f"What color is this {object_name}?"
                
            prompt = f"{instruction_tokens} Answer with one word. {question} {end_tokens}"    

        print(prompt)

        # --- Model-specific input processing ---
        if model_version == "janus":
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{prompt}",
                    "images": [image],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]
            pil_images = load_pil_images(conversation)
            prepare_inputs = processor(
                conversations=conversation, images=pil_images, force_batchify=True
            ).to(model.device)
            inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
            outputs = model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                output_attentions=True,
                use_cache=True
            )
            input_ids = prepare_inputs["input_ids"].detach().cpu().numpy()[0]
            tokenizer = processor.tokenizer

        elif model_version == "llava-next":
            inputs = processor(images=image, text=prompt, return_tensors='pt')
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            outputs = model(**inputs, output_attentions=True)
            input_ids = inputs["input_ids"].detach().cpu().numpy()[0]
            tokenizer = processor.tokenizer

        elif model_version == "qwen":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{image_path}"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            outputs = model(**inputs, output_attentions=True)
            input_ids = inputs["input_ids"].detach().cpu().numpy()[0]
            tokenizer = processor.tokenizer

        # --- Get token indices ---
        img_indices, text_indices = get_image_token_indices(model_version, input_ids, tokenizer)

        # --- Attention mass computation ---
        for layer_idx in range(len(outputs.attentions)):
            attn = outputs.attentions[layer_idx].squeeze().detach().cpu().to(torch.float32).numpy()
            attn_from_last_token = attn[:, -1, :]

            attn_to_img = attn_from_last_token[:, img_indices]
            img_mass = np.sum(attn_to_img) / attn_to_img.shape[0] if attn_to_img.ndim == 2 else float(attn_to_img)

            attn_to_text = attn_from_last_token[:, text_indices]
            text_mass = np.sum(attn_to_text) / attn_to_text.shape[0] if attn_to_text.ndim == 2 else float(attn_to_text)

            entry[f"img_mass_layer_{layer_idx}"] = img_mass
            entry[f"text_mass_layer_{layer_idx}"] = text_mass

        results[f"{most}_{idx}"] = dict(entry)

        # --- Cleanup ---
        for var in ['outputs', 'inputs', 'image', 'prepare_inputs']:
            if var in locals():
                del locals()[var]
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

    return results
    
"""
def run_attention_mass_analysis(batch_df, task, processor, model, model_version, image_type, most, instruction_tokens, end_tokens):
    results = {}

    for idx, row in batch_df.iterrows():
        entry = defaultdict(list)

        # --- Image path ---
        if task == "size":
            image_path = row['path_to_counterfact'] if image_type == 'counterfact' else row['path_to_clean']
        else:
            base_path = "/oscar/data/ceickhof/visual_counterfact/"
            if image_type == 'counterfact':
                color_to_replace = row['image_path'].split('_')[1]
                new_path = row['image_path'].replace(color_to_replace, row['incorrect_answer'])
                image_path = new_path.replace('downloaded_images', 'downloaded_images_counterfact')
                image_path = base_path + "final_counterfact_images/" + image_path
            else:
                image_path = row['image_path']

        # --- Load image ---
        try:
            if model_version == "janus":
                with open(image_path, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode("utf-8")
                    image = f"data:image/jpeg;base64,{image_data}"
            else:
                image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Image not found at {image_path}")
            continue

        # --- Build prompt ---
        if task == "size":
            prompt = row["prompt_most"] if most == "True" else row["prompt_this"]
            prompt = f"{instruction_tokens} {prompt} {end_tokens}"
        else:
            if most == "True":
                prompt = f"{instruction_tokens} Answer with one word. What color is a {row['correct_object']}? {end_tokens}"
            else:
                prompt = f"{instruction_tokens} Answer with one word. What color is this {row['correct_object']}? {end_tokens}"

        print(prompt)

        # --- Prepare inputs and run forward pass ---
        if model_version == "janus":
            conversation = [
                {
                    "role": "<|User|>",
                    "content": f"<image_placeholder>\n{prompt}",
                    "images": [image],
                },
                {"role": "<|Assistant|>", "content": ""},
            ]

            pil_images = load_pil_images(conversation)
            prepare_inputs = processor(
                conversations=conversation, images=pil_images, force_batchify=True
            ).to(model.device)

            inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
            outputs = model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                output_attentions=True,
                use_cache=True
            )
            input_ids = prepare_inputs["input_ids"].detach().cpu().numpy()[0]
            tokenizer = processor.tokenizer
            img_token_id = tokenizer.convert_tokens_to_ids("<image>")

        elif model_version == "llava-next":
            if task == "color":
                image = image.resize((256, 256), Image.LANCZOS if task == "color" else Image.LANCZOS)
            else:
                image = image.resize((256, 200), Image.LANCZOS if task == "color" else Image.LANCZOS)
            inputs = processor(images=image, text=prompt, return_tensors='pt')
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            outputs = model(**inputs, output_attentions=True)
            input_ids = inputs["input_ids"].detach().cpu().numpy()[0]
            img_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")

        elif model_version == "qwen":
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{image_path}"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")
            outputs = model(**inputs, output_attentions=True)
            input_ids = inputs["input_ids"].detach().cpu().numpy()[0]
            img_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")

        # --- Token index separation ---
        img_indices = np.where(input_ids == img_token_id)
        text_indices = np.where(input_ids != img_token_id)

        # --- Attention mass per layer ---
        for layer_idx in range(len(outputs.attentions)):
            attn = outputs.attentions[layer_idx].squeeze().detach().cpu().to(torch.float32).numpy()
            attn_from_last_token = attn[:, -1, :]  # shape: [heads, tokens]

            attn_to_img = attn_from_last_token[:, img_indices].squeeze(axis=1)
            attn_to_text = attn_from_last_token[:, text_indices].squeeze(axis=1)

            entry[f"img_mass_layer_{layer_idx}"] = np.sum(attn_to_img) / attn_to_img.shape[0]
            entry[f"text_mass_layer_{layer_idx}"] = np.sum(attn_to_text) / attn_to_text.shape[0]

        results[f"{most}_{idx}"] = dict(entry)

        # --- Cleanup ---
        del outputs
        if 'inputs' in locals(): del inputs
        if 'image' in locals(): del image
        if 'prepare_inputs' in locals(): del prepare_inputs
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

    return results
"""
def main():
    # TODO: update all file-names and paths to run on multiple tasks 
    # image_type = "counterfact" most="True"
    parser = argparse.ArgumentParser(description="Run MLLMs on all tasks.")
    parser.add_argument('--model_version', type=str, choices=['llava-next', 'qwen' ,'janus'], required=True, help="Choose the model version.")
    parser.add_argument('--task', type=str, choices=['color', 'size'], required=True, help="Choose the task.")
    parser.add_argument('--dataset_size', type=str, choices=['mini', 'full'], required=True, help="Choose dataset size (mini or full).")
    parser.add_argument('--image_type', type=str, choices=['counterfact', 'real'], required=True, help="Choose image type.")
    parser.add_argument('--batch_size', type=int, default=50, help="Batch size for inference (default: 50)")
    #parser.add_argument('--most', type=str, choices=['True', 'False'], required=True, help="Choose if using 'this' or 'most'.")

    args = parser.parse_args()

    print(torch.cuda.is_available())
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    if args.model_version == 'llava-next':
        processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf",quantization_config=bnb_config, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        
        
    elif args.model_version == 'janus':
        model_path = "deepseek-ai/Janus-Pro-7B"
        processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
        model = model.to(torch.bfloat16).cuda().eval()

    elif args.model_version == 'qwen':
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
        )

    # load DF. 
    """
    if args.task == "color":
        df = pd.read_csv("/oscar/data/ceickhof/visual_counterfact/final_images_with_counterfact.csv") #.head(10)
        
    elif args.task == "size":
        df = pd.read_csv("with_line_sizes_dup.csv")
    """
    
    if args.task == "color":

        if args.model_version == "llava-next":
            df = pd.read_csv("color_filtered_WK.csv") #.head(5)
            
        elif args.model_version == "qwen":
            df = pd.read_csv("color_qwen_filtered_WK.csv")
            
        elif args.model_version == "janus":
            df = pd.read_csv("color_janus_filtered_WK.csv") #.head(5)
    
            
    elif args.task == "size":
        #df = pd.read_csv("with_line_sizes_dup.csv") #.head(10)
        
        #df = pd.read_csv("with_line_sizes_balanced.csv")
        
        #df = pd.read_csv("size_filtered_WK.csv")

        if args.model_version == "llava-next":
            df = pd.read_csv("size_df_for_TV_llava-next.csv") #.head(5)
            
        elif args.model_version == "qwen":
            df = pd.read_csv("size_df_for_TV_qwen.csv")
            
        elif args.model_version == "janus":
            df = pd.read_csv("size_df_for_TV_janus.csv") #.head(5)
    
    
    batch_size = args.batch_size
    attention_results = {}
    
    for most in ["True", "False"]:
        print(f"\nProcessing most={most}")
    
        for i in range(0, len(df), batch_size):
            print(f"  Batch {i} → {i + batch_size}")
            batch_df = df.iloc[i:i+batch_size]
    
            with torch.inference_mode():
                batch_result = run_attention_mass_analysis(
                    batch_df=batch_df,
                    task = args.task,
                    processor=processor,
                    model=model,
                    model_version = args.model_version,
                    image_type="counterfact",
                    most=most,
                    instruction_tokens="[INST] <image>\n" if args.model_version == "llava-next" else "",
                    end_tokens="[/INST]" if args.model_version == "llava-next" else ""
                )
    
            attention_results.update(batch_result)
    
            # Cleanup
            del batch_df
            del batch_result
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
    
        
    #answers = mllm_early_decoding(df, processor, model, args.model_version, args.task,  most=args.most)

    file_path = f'attention_mass_{args.task}_{args.model_version}_{len(df)}.pickle'

    with open(file_path, "wb") as f:
        pickle.dump(attention_results, f)
    
if __name__ == "__main__":
    main()