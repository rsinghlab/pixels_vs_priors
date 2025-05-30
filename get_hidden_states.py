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

from datasets import load_dataset
import io
import random 

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

import torch
import numpy as np
from PIL import Image
import gc
from collections import defaultdict

def get_hidden_states(batch_df, task,  processor, model, model_name, image_type, most, instruction_tokens, end_tokens):
    states = {}
    
    for idx, row in batch_df.iterrows():

        if task == "color":
            if image_type == 'counterfact':
                image_path = row['counterfact_image']['bytes']
            else:
                image_path = row['original_image']['bytes']
                
            object_name = row['object']
            #question = f"What color is {'a' if most == 'True' else 'this'} {object_name}?"
            if most == "True":
                object_name_plural = object_name if object_name.endswith("s") else object_name + "s"
                question = f"What color are most {object_name_plural}?"
                
            else:
                question = f"What color is this {object_name}?"
                
            prompt = f"{instruction_tokens} Answer with one word. {question} {end_tokens}"    
            
        elif task == "size":
            if image_type == "counterfact":
                image_path = row['counterfact_image']['bytes']
            else:
                image_path = row['original_image']['bytes']
            
            # Randomly select whether the correct/incorrect object goes first. 
            if most == "True":
                objects = [row['correct_answer'], row['incorrect_answer']]
                random.shuffle(objects)
                prompt = f"Answer with the correct option. Which is larger usually, {objects[0]} or {objects[1]}?"
            else:
                objects = [row['correct_answer'], row['incorrect_answer']]
                random.shuffle(objects)
                prompt = f"Answer with the correct option. Which is larger here, {objects[0]} or {objects[1]}?" 
            prompt = f"{instruction_tokens} {prompt} {end_tokens}"
            
        try:
            image = Image.open(io.BytesIO(image_path)).convert("RGB")
            image = image.resize((256, 200) if task == "size" else (256, 256), Image.LANCZOS)
        except FileNotFoundError:
            print(f"Warning: Image not found for {row['object']}")
            continue  # Skip to the next row in the DataFrame 

        # --- Model forward ---
        #inputs = processor(images=image, text=prompt, return_tensors='pt')
        #inputs = {k: v.to('cuda') for k, v in inputs.items()}

        #outputs = model(**inputs, output_hidden_states=True)

        if model_name == "janus":
             image_data = base64.b64encode(image_path).decode("utf-8")
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
             outputs = model.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=10,
                do_sample=False,
                use_cache=False,
                output_hidden_states=True,
            )

        elif model_name == 'llava-next':   
            inputs = processor(images=image, text=prompt, return_tensors='pt')
            inputs = {k: v.to(model.device) for k, v in inputs.items()} 
            # Perform a forward pass with the model
            #model = model.cuda()
            outputs = model(**inputs, output_hidden_states=True)  # Adjust max_new_tokens as needed
            
        elif model_name == 'qwen':
            pil_img = Image.open(io.BytesIO(image_path)).convert("RGB") 
            pil_img = pil_img.resize((224, 224), Image.LANCZOS)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": pil_img,
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Preparation for inference
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
            inputs = inputs.to("cuda")
            torch.cuda.empty_cache()
            outputs = model(**inputs, output_hidden_states=True)
        
        if len(states) == 0:
            for i in range(len(outputs.hidden_states)):
                states[i] = []
        
        for i in range(len(outputs.hidden_states)):
            with torch.no_grad():
                states[i].append(outputs['hidden_states'][i][:,-1,:].detach().cpu().to(torch.float32).numpy())
            torch.cuda.empty_cache()
        

        # --- Cleanup ---
        del outputs
        if 'inputs' in locals(): del inputs
        if 'image' in locals(): del image
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

    return states

def main():
    # TODO: update all file-names and paths to run on multiple tasks 
    # image_type = "counterfact" most="True"
    parser = argparse.ArgumentParser(description="Run MLLMs on all tasks.")
    parser.add_argument('--model_version', type=str, choices=['llava-next', 'qwen' ,'janus'], required=True, help="Choose the model version.")
    parser.add_argument('--task', type=str, choices=['color', 'size'], required=True, help="Choose the task.")
    parser.add_argument('--dataset_size', type=str, choices=['mini', 'full'], required=True, help="Choose dataset size (mini or full).")
    parser.add_argument('--image_type', type=str, choices=['counterfact', 'real'], required=True, help="Choose image type.")
    parser.add_argument('--batch_size', type=int, default=50, help="Batch size for inference (default: 50)")
    parser.add_argument('--most', type=str, choices=['True', 'False'], required=True, help="Choose if using 'this' or 'most'.")

    args = parser.parse_args()
    random.seed(0)
    
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

    dataset = load_dataset("mgolov/Visual-Counterfact")

    # load DF. 
    if args.task == "color":
        df = dataset["color"].to_pandas()
        if args.dataset_size == "mini":
            df = df.head(5)
        
        
    elif args.task == "size":
        if args.line == "True":
            df = dataset["size"].to_pandas()
        if args.dataset_size == "mini":
            df = df.head(5) 

    
    batch_size = args.batch_size
    all_hidden_states = {}  # Will store final hidden state arrays per layer
    
    print(f"\nProcessing most={args.most}")
    
    for i in range(0, len(df), batch_size):
        print(f"  Batch {i} → {i + batch_size}")
        batch_df = df.iloc[i:i+batch_size]
    
        with torch.inference_mode():
            batch_result = get_hidden_states(
                batch_df=batch_df,
                task=args.task,
                processor=processor,
                model=model,
                model_name = args.model_version,
                image_type="counterfact",
                most=args.most,
                instruction_tokens="[INST] <image>\n" if args.model_version == "llava-next" else "",
                end_tokens="[/INST]" if args.model_version == "llava-next" else ""
            )
    
        # Initialize keys on first batch
        if not all_hidden_states:
            all_hidden_states = {k: [] for k in batch_result}
    
        # Append batch results
        for k in all_hidden_states:
            all_hidden_states[k].extend(batch_result[k])
    
        # Cleanup
        del batch_df, batch_result
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
    
    # Stack once all batches are done
    for k in all_hidden_states:
        all_hidden_states[k] = np.vstack(all_hidden_states[k])
    
    # Save to file
    file_path = f'hidden_states_{args.task}_{args.model_version}_{len(df)}_most_{args.most}.pickle'
    with open(file_path, "wb") as f:
        pickle.dump(all_hidden_states, f)
    
if __name__ == "__main__":
    main()