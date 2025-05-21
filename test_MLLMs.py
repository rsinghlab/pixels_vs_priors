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

from PIL import Image
import matplotlib.pyplot as plt

from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig

# JANUS IMPORTS
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
import base64

import argparse
import os
import pickle
import gc

from collections import defaultdict
from qwen_vl_utils import process_vision_info

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def clean_janus_answer(text):
    pattern = r"<\|Assistant\|>:\s?(.*?)(?=<｜end▁of▁sentence｜>)"
    match = re.search(pattern, text)
    return match.group(1).strip('.')

def clean_instruction_tokens(text):
    cleaned_text = re.sub(r'\[INST\]\s*\n?.*?\[/INST\]\s*', '', text, flags=re.DOTALL)
    return cleaned_text.strip()


def mllm_testing(df, processor, model, model_name, task, image_type, most="True"):
    with torch.inference_mode():
        torch.cuda.empty_cache()
        gc.collect()
        generated_texts = []
        for idx, row in df.iterrows():
    
            if model_name == 'llava-next':
                instruction_tokens = "[INST] <image>\n"
                end_tokens = "[/INST]"
            else:
                instruction_tokens=''
                end_tokens=''
                
            if task == "color":
                base_path = "/oscar/data/ceickhof/visual_counterfact/"
                if image_type == 'counterfact':
                    color_to_replace = row['image_path'].split('_')[1]
                    new_path = row['image_path'].replace(color_to_replace, row['incorrect_answer'])
                    image_path = new_path.replace('downloaded_images', 'downloaded_images_counterfact')
                    image_path = base_path + "final_counterfact_images/" + image_path
                else:
                    image_path = row['image_path']
                    
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
                
            elif task == "size":
                if image_type == "counterfact":
                    image_path = row['path_to_counterfact']
                else:
                    image_path = row['path_to_clean']
                
                prompt = row["prompt_most"] if most == "True" else row["prompt_this"]
                prompt = f"{instruction_tokens} {prompt} {end_tokens}"
                print(prompt)
                
            
            try:
                image = Image.open(image_path).convert("RGB")
                image = image.resize((256, 200) if task == "size" else (256, 256), Image.LANCZOS)
            except FileNotFoundError:
                print(f"Warning: Image not found at {image_path}")
                generated_texts.append(None)
                continue  # Skip to the next row in the DataFrame
    
            if model_name == "janus":
                 with open(image_path, "rb") as image_file:
                    image_data = base64.b64encode(image_file.read()).decode("utf-8")
                    image = f"data:image/jpeg;base64,{image_data}"
    
                 conversation = [
                    {
                        "role": "<|User|>",
                        "content": f"<image_placeholder>\n{prompt}",
                        "images": [image],
                    },
                    {"role": "<|Assistant|>", "content": ""},
                ]
    
                 # load images and prepare for inputs
                 pil_images = load_pil_images(conversation)
    
                 prepare_inputs = processor(
                    conversations=conversation, images=pil_images, force_batchify=True
                 ).to(model.device)
    
                 # # run image encoder to get the image embeddings
                 inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)
                 tokenizer = processor.tokenizer
    
                 # removing 'generate' for now. 
                 outputs = model.language_model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=prepare_inputs.attention_mask,
                    pad_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_new_tokens=10,
                    num_beams=1,
                    do_sample=False,
                    use_cache=True,
                    temperature=1.0,
                )
                 predicted_answer = processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
                
    
            elif model_name == 'llava-next':   
                inputs = processor(images=image, text=prompt, return_tensors='pt')
                inputs = {k: v.to('cuda') for k, v in inputs.items()} 
                # Perform a forward pass with the model
                outputs = model.generate(**inputs, max_new_tokens=10, num_beams=1, do_sample=False, temperature=1.0)  # Adjust max_new_tokens as needed
                predicted_answer = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
                predicted_answer = clean_instruction_tokens(predicted_answer)
                
            elif model_name == 'qwen':
                pil_img = Image.open(image_path).convert("RGB")
                pil_img = pil_img.resize((224, 224), Image.LANCZOS)
                
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},  # Pass resized image object
                        {"type": "text", "text": prompt},
                    ],
                }]
                """
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{image_path}"},
                        {"type": "text", "text": prompt},
                    ],
                }]
                """
    
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
    
                image_inputs, video_inputs = process_vision_info(messages)
    
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
    
                for k, v in inputs.items():
                    if torch.is_tensor(v):
                        inputs[k] = v.to("cuda", non_blocking=True)
    
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    use_cache=False
                )
    
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
                ]
    
                output_text = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
    
                predicted_answer = output_text[0]
    
                # Explicit cleanup
                del image_inputs, video_inputs, inputs, generated_ids
                #model.cpu()
                torch.cuda.empty_cache()
                gc.collect()
                
                   
            
            generated_texts.append(predicted_answer)
            #print(torch.cuda.memory_summary())
            
            to_delete = ['inputs', 'outputs', 'image_inputs', 'video_inputs', 'generated_ids', 'prepare_inputs', 'image', 'pil_images', 'inputs_embeds']
            for var_name in to_delete:
                if var_name in locals():
                    var = locals()[var_name]
                    if isinstance(var, dict):
                        for v in var.values():
                            if torch.is_tensor(v) and v.is_cuda:
                                del v
                    elif torch.is_tensor(var) and var.is_cuda:
                        del var
                    del locals()[var_name]
            if hasattr(model, 'clear_kv_cache'):
                model.clear_kv_cache()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            gc.collect()
            
            #print(torch.cuda.memory_summary())
                
    
        df['generated_text'] = generated_texts
    
        if 'inputs' in locals(): del inputs
        if 'image' in locals(): del image
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
    return df

def main():
    # TODO: update all file-names and paths to run on multiple tasks 
    # image_type = "counterfact" most="True"
    parser = argparse.ArgumentParser(description="Run MLLMs on all tasks.")
    parser.add_argument('--model_version', type=str, choices=['llava-next', 'qwen' ,'janus'], required=True, help="Choose the model version.")
    parser.add_argument('--task', type=str, choices=['color', 'size'], required=True, help="Choose the task.")
    parser.add_argument('--line', type=str, choices=['True', 'False'], required=True, help="Only for size.")
    parser.add_argument('--dataset_size', type=str, choices=['mini', 'full'], required=True, help="Choose dataset size (mini or full).")
    parser.add_argument('--image_type', type=str, choices=['counterfact', 'real'], required=True, help="Choose image type.")
    parser.add_argument('--most', type=str, choices=['True', 'False'], required=True, help="Choose if using 'this' or 'most'.")

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
            "llava-hf/llava-v1.6-mistral-7b-hf", quantization_config=bnb_config, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        
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
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto"
        )
        model.eval()
        model = model.to(torch.bfloat16).cuda().eval()
        #model.cuda()

    # load DF. 
    if args.task == "color":
        df = pd.read_csv("/oscar/data/ceickhof/visual_counterfact/final_images_with_counterfact.csv") #.head(10)
        if args.dataset_size == "mini":
            df = df.head(5)
        
        
    elif args.task == "size":
        if args.line == "True":
            df = pd.read_csv("with_line_sizes_dup.csv")
            #df = pd.read_csv("with_line_sizes_dup_plural.csv")
            print("with line!")
        else:
            df = pd.read_csv("no_line_sizes.csv")
            print("no line!")
            
        #df = pd.read_csv("composite_images.csv")
        if args.dataset_size == "mini":
            df = df.head(5)
    
        
    #df = mllm_testing(df, processor, model, args.model_version, args.task, args.image_type, most=args.most)
    batch_size = 1
    results = []
    
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i + batch_size].copy()
        with torch.inference_mode():
            result_df = mllm_testing(batch_df, processor, model, args.model_version, args.task, args.image_type, most=args.most)
            
        results.append(result_df)
        del result_df
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
    
    df = pd.concat(results, ignore_index=True)

    df.to_csv(f'most_instances_plural_bigger_{args.task}_new_MLLM_results_most_{args.most}_{args.image_type}_line_{args.line}_{args.model_version}_{args.dataset_size}.csv', index=False)

    
if __name__ == "__main__":
    main()