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
import random
from datasets import load_dataset
import io

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

def get_lm_preds(df, processor, model, model_name, task):
    incorrect_idx_ex = []
    incorrect_idx = []
    preds = []
    
    for idx, row in df.iterrows():
        if model_name == 'llava-next': 
            instruction_tokens = "[INST]\n"
            end_tokens = "[/INST]"
        
        elif model_name == 'janus':
            instruction_tokens = "\n<|User|>:"
            end_tokens = "\n<|Assistant|>:"
            #prompt = f"\n<|User|>: Answer with one word. What color are most {row['correct_object']}? \n<|Assistant|>:"
         
        elif model_name == 'qwen':
            #prompt = f"Answer with one word. What color are most {row['correct_object']}"
            instruction_tokens = ""
            end_tokens = ""
            
        if task == "color":    
            #prompt = f"{instruction_tokens} Answer with one word. What color are most {row['correct_object']}s? {end_tokens}"
            object_name = row['object']
            object_name_plural = object_name if object_name.endswith("s") else object_name + "s"
            question = f"What color are most {object_name_plural}?"
            prompt = f"{instruction_tokens} Answer with one word. {question} {end_tokens}"   
                
            
        elif task == "size":            
            # Randomly select whether the correct/incorrect object goes first. 
            objects = [row['correct_answer'], row['incorrect_answer']]
            random.shuffle(objects)
            prompt = f"Answer with the correct option. Which is larger usually, {objects[0]} or {objects[1]}?"
            prompt = f"{instruction_tokens} {prompt} {end_tokens}" 

        inputs = processor.tokenizer(text=prompt, return_tensors='pt').to(model.device)

        # Perform a forward pass with the model
        if model_name == 'llava-next':
            outputs = model.language_model.generate(**inputs, max_new_tokens=10, num_beams=1, do_sample=False, temperature=1.0, pad_token_id=processor.tokenizer.pad_token_id)  # Adjust max_new_tokens as needed
            predicted_answer = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
            predicted_answer = clean_instruction_tokens(predicted_answer)

        elif model_name == 'janus':
            outputs = model.language_model.generate(**inputs,
                pad_token_id=processor.tokenizer.eos_token_id,
                bos_token_id=processor.tokenizer.bos_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                max_new_tokens=10, num_beams=1, do_sample=False, temperature=1.0)
            # Keeping special tokens bc it makes the regex easier to write. 
            predicted_answer = clean_janus_answer(processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=False))     
            
        
        elif model_name == 'qwen':
            conversation = [
                {
                    "role": 'user',
                    'content': [
                        {'type': 'text', 'text': prompt}
                    ],
                },
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(
                text=prompt,
                return_tensors='pt',
            ).to(model.device, torch.float16 if model.device.type == 'cuda' else torch.float32)
            
            outputs = model.generate(**inputs, max_new_tokens=10, num_beams=1, do_sample=False, temperature=1.0)
            predicted_answer = processor.batch_decode(outputs, skip_special_tokens=True)[0].split('\nassistant\n')[-1]

        
        pred = predicted_answer.lower().strip()
        
        if pred == 'gray':
            pred='grey'

        preds.append(pred)
        
        if row["correct_answer"].lower() not in pred: #not in row["all_colors"]:
            print("Prompt: ", prompt)
            print("PRED: ", pred)
            print("Label: ", row["correct_answer"])
            incorrect_idx_ex.append(row.index)


    print("EXACT MATCH ACC: ",(len(preds) - len(incorrect_idx_ex))/len(preds)*100 )
    
    if task == "color":
        print("MCRAE VALID ACC: ", (len(preds) - len(incorrect_idx))/len(preds)*100)
    
    np.save(f"{task}_preds_{model_name}.npy", np.array(preds))
    df[f'{model_name}_lm_preds'] = preds 
    df.to_csv(f"{task}_preds_{model_name}.csv",index=False)

    return df


def mllm_early_decoding(df, processor, model, model_name, task, most="True"):
    answers = {}
    for idx, row in df.iterrows():

        if model_name == 'llava-next':
            instruction_tokens = "[INST] <image>\n"
            end_tokens = "[/INST]"
        else:
            instruction_tokens=''
            end_tokens=''
            
        if task == "color":
            image_path = row['counterfact_image']['bytes']                    
            object_name = row['object']
            #question = f"What color is {'a' if most == 'True' else 'this'} {object_name}?"
            if most == "True":
                object_name_plural = object_name if object_name.endswith("s") else object_name + "s"
                question = f"What color are most {object_name_plural}?"
                
            else:
                question = f"What color is this {object_name}?"
                
            prompt = f"{instruction_tokens} Answer with one word. {question} {end_tokens}" 
        
        elif task == "size":
            image_path = row['counterfact_image']['bytes']

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
                max_new_tokens=10, num_beams=1, do_sample=False, temperature=1.0,
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
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"file://{image_path}",
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

        
        answers[idx] = defaultdict(str)
        answers[idx]["counterfact_answer"] = row["incorrect_answer"]
        answers[idx]["correct_answer"] = row["correct_answer"]
        
        
        answers[idx]["early_preds"] = []
        answers[idx]["prob_counterfact"] = []
        answers[idx]["prob_correct"] = [] 
        answers[idx]["max_prob"] = []

        if task == "color":
            answers[idx]["lm_pred_answer"] = row[f'{model_name}_lm_preds']
            answers[idx]["prob_lm_pred"] = [] 
        

        #answers[idx]["best_correct_alternative"] = []
        #answers[idx]["prob_correct_alternative"] = []
        #answers[idx]["best_counterfact_alternative"] = []
        #answers[idx]["prob_counterfact_alternative"] = []

        
        # Iterate over hidden states and apply the LM-Head
        for i in range(len(outputs.hidden_states)):
            with torch.no_grad():
                torch.cuda.empty_cache()
                if model_name == 'qwen':
                    early_output = model.model.norm(outputs['hidden_states'][i][:,-1,:].unsqueeze(dim=0))
                    logits = model.lm_head(early_output)
                else:
                    #print(get_nvidia_smi())
                    early_output = model.language_model.model.norm(outputs['hidden_states'][i][:,-1,:].unsqueeze(dim=0))
                    logits = model.language_model.lm_head(early_output) 
                
                pred_token = logits.argmax(dim=-1)  # Get predicted tokens
                early_decoded = processor.tokenizer.decode(pred_token[0], skip_special_tokens=True)

                answers[idx]["early_preds"].append(early_decoded)        
                softmax = torch.nn.functional.softmax(logits.squeeze().to(torch.float16), dim=-1).detach().cpu().numpy()

            # double check this is correct still. 
            token_idx = 0
            if model_name in ['janus', 'llava-next']:
                token_idx = 1

            correct_idx = processor.tokenizer(row["correct_answer"]).input_ids[token_idx]
            correct_idx_cap = processor.tokenizer(row["correct_answer"].capitalize()).input_ids[token_idx]
            
            correct_prob = max(softmax[correct_idx], softmax[correct_idx_cap])

            if task == "color":
                pred_idx = processor.tokenizer(row[f'{model_name}_lm_preds']).input_ids[token_idx]
                pred_idx_cap = processor.tokenizer(row[f'{model_name}_lm_preds'].capitalize()).input_ids[token_idx]
                lm_pred_prob = max(softmax[pred_idx], softmax[pred_idx_cap])
            
            counterfact_idx = processor.tokenizer(row["incorrect_answer"]).input_ids[token_idx]
            counterfact_idx_cap = processor.tokenizer(row["incorrect_answer"].capitalize()).input_ids[token_idx]
            counterfact_prob = max(softmax[counterfact_idx], softmax[counterfact_idx_cap])

            
            answers[idx]["prob_counterfact"].append(counterfact_prob)
            answers[idx]["prob_correct"].append(correct_prob)
            answers[idx]["max_prob"].append(np.max(softmax))

            if task == "color": 
                answers[idx]["prob_lm_pred"].append(lm_pred_prob)

            #answers[idx]["best_correct_alternative"].append(best_correct_alternative if best_correct_alternative else "None")
            #answers[idx]["prob_correct_alternative"].append(best_correct_alt_prob if best_correct_alt_prob is not None else "None")
        
            #answers[idx]["best_counterfact_alternative"].append(best_counterfact_alternative if best_counterfact_alternative else "None")
            #answers[idx]["prob_counterfact_alternative"].append(best_counterfact_alt_prob if best_counterfact_alt_prob is not None else "None")
        del outputs
        if 'inputs' in locals():
            del inputs
        if 'image' in locals():
            del image
        if 'image_inputs' in locals():
            del image_inputs
        if 'prepare_inputs' in locals():
            del prepare_inputs
        if 'early_output' in locals():
            del early_output
        if 'logits' in locals():
            del logits
        if 'softmax' in locals():
            del softmax
        
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        
    return answers

def main():
    # TODO: update all file-names and paths to run on multiple tasks 
    # image_type = "counterfact" most="True"
    parser = argparse.ArgumentParser(description="Run MLLMs on all tasks.")
    parser.add_argument('--model_version', type=str, choices=['llava-next', 'qwen' ,'janus'], required=True, help="Choose the model version.")
    parser.add_argument('--task', type=str, choices=['color', 'size'], required=True, help="Choose the task.")
    parser.add_argument('--dataset_size', type=str, choices=['mini', 'full'], required=True, help="Choose dataset size (mini or full).")
    parser.add_argument('--image_type', type=str, choices=['counterfact', 'real'], required=True, help="Choose image type.")
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

    # load DF. 
    
    dataset = load_dataset("mgolov/Visual-Counterfact")
    
    if args.task == "color":
        df = dataset["color"].to_pandas()
        if not os.path.exists(f"{args.task}_preds_{args.model_version}.npy"):
            # Your code block here
            print("Generating LM-Only Predictions")
            df = get_lm_preds(df, processor, model, args.model_version, args.task)
            
        else:
            # If we have the LM-only preds, load them into the df. 
            preds = np.load(f"{args.task}_preds_{args.model_version}.npy")
            df[f'{args.model_version}_lm_preds'] = preds
        
    elif args.task == "size":
        """
        mid = len(df) // 2
        top_half = df.iloc[:mid]
        bottom_half = df.iloc[mid:]
        """
        df = dataset["color"].to_pandas() 
        
        if not os.path.exists(f"{args.task}_preds_{args.model_version}.npy"):
            # Your code block here
            print("Generating LM-Only Predictions")
            df = get_lm_preds(df, processor, model, args.model_version, args.task)
            
        else:
            # If we have the LM-only preds, load them into the df. 
            preds = np.load(f"{args.task}_preds_{args.model_version}.npy")
            df[f'{args.model_version}_lm_preds'] = preds
               
    batch_size = 50
    answers = {}

    if args.dataset_size == 'mini':
        df = df.head(5)
    
    for i in range(0, len(df), batch_size):
        print(f"Processing batch {i} → {i + batch_size}")
    
        batch_df = df.iloc[i:i+batch_size]
        
        with torch.inference_mode():
    
            batch_answers = mllm_early_decoding(
                batch_df, processor, model,
                args.model_version, args.task, most=args.most
            )
    
        # Merge dictionaries
        answers.update(batch_answers)
    
        # Cleanup
        del batch_df
        del batch_answers
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
    
        
    #answers = mllm_early_decoding(df, processor, model, args.model_version, args.task,  most=args.most)

    if args.most == "True":
        file_path = f'most_new_{args.task}_early_decoding_results_most_{args.image_type}_{args.model_version}.pickle'

    if args.most == "False":
        file_path = f'new_{args.task}_early_decoding_results_{args.image_type}_{args.model_version}.pickle'
        
    with open(file_path, 'wb') as f:
        pickle.dump(answers, f)

    
if __name__ == "__main__":
    main()