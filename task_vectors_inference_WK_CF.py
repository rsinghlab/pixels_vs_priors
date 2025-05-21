import torch

torch.cuda.empty_cache()  # Frees unused memory
torch.cuda.reset_peak_memory_stats()  # Resets memory stats (optional)

import torch.nn as nn
from datasets import load_dataset
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

# JANUS IMPORTS
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
import base64

from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
from IPython.display import display
import pickle 
from qwen_vl_utils import process_vision_info
import argparse

def get_outliers(states):
    # Calculate Outliers in the 0-shot setting
    var_per_dimension = np.var(states, axis=0)
    
    # Step 2: Compute the average STD
    avg_std = np.mean(var_per_dimension)
    
    # Step 3: Identify indices where Variance is 5 times larger than the average variance
    indices = np.where(var_per_dimension >= 5 * avg_std)[0]
    large_var_values = var_per_dimension[indices]
    
    # Step 4: Print or return the results
    result = list(zip(indices, large_var_values))
    print("Indices and Var values for dimensions with Var >= 5x average STD:")
    for idx, var_val in result:
        print(f"Dimension {idx}: Var = {var_val}")
        
    return [x[0] for x in sorted(result, key=lambda x: x[1], reverse=True)]

def clean_instruction_tokens(text):
    cleaned_text = re.sub(r'\[INST\]\s*\n?.*?\[/INST\]\s*', '', text, flags=re.DOTALL)
    return cleaned_text.strip()


def get_answers(model, model_name, task, processor, df, image_type, most, instruction_tokens, end_tokens):
    
    #base_path = "../multimodal_MI/visual_counterfact/"
    base_path = "/oscar/data/ceickhof/visual_counterfact/"


    answers = {}
    for idx, row in df.iterrows():
        if task == "size":
            # --- Image loading ---
            image_path = row['path_to_counterfact'] if image_type == 'counterfact' else row['path_to_clean']
            try:
                image = Image.open(image_path).convert("RGB").resize((256, 200), Image.LANCZOS)
            except FileNotFoundError:
                print(f"Warning: Image not found at {image_path}")
                continue
    
            # --- Prompt construction ---
            prompt = row["prompt_most"] if most == "True" else row["prompt_this"]
            prompt = f"{instruction_tokens} {prompt} {end_tokens}"
            #print(prompt)
            
        else:
            base_path = "/oscar/data/ceickhof/visual_counterfact/"
            if image_type == 'counterfact':
                color_to_replace = row['image_path'].split('_')[1]
                new_path = row['image_path'].replace(color_to_replace, row['incorrect_answer'])
                image_path = new_path.replace('downloaded_images', 'downloaded_images_counterfact')
                image_path = base_path + "final_counterfact_images/"  + image_path
            elif image_type == 'clean': 
                image_path = row['image_path']
                 
            try:
               image = Image.open(image_path)
            except FileNotFoundError:
               print(f"Warning: Image not found at {image_path}")
               continue  # Skip to the next row in the DataFrame
        
            # Resize images to 256 x 256. This is NEEDED for the google images
            image = image.resize((256, 256), Image.LANCZOS)
            """
            if most == "True":
                prompt = f"{instruction_tokens} Answer with one word. What color is a {row['correct_object']}? {end_tokens}"
            else:
                prompt = f"{instruction_tokens} Answer with one word. What color is this {row['correct_object']}? {end_tokens}"
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
            #print(prompt)
            #print(prompt)
        """    
        inputs = processor(images=image, text=prompt, return_tensors='pt')
        inputs = {k: v.to('cuda') for k, v in inputs.items()} 
        
        # Perform a forward pass with the model
        outputs = model.generate(**inputs, max_new_tokens=20, pad_token_id=processor.tokenizer.pad_token_id, do_sample=False, num_beams=1, temperature =1.0)  # Adjust max_new_tokens as needed
        predicted_answer = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        predicted_answer = clean_instruction_tokens(predicted_answer)
        """
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
                max_new_tokens=20,
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
            outputs = model.generate(**inputs, max_new_tokens=20, pad_token_id=processor.tokenizer.pad_token_id, do_sample=False, num_beams=1, temperature =1.0)  # Adjust max_new_tokens as needed
            predicted_answer = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
            predicted_answer = clean_instruction_tokens(predicted_answer)
            
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
            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, max_new_tokens=20, num_beams=1, do_sample=False, temperature=1.0)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            
            predicted_answer = output_text[0]

        
        if task == "color":
            answers[row['correct_object']] = predicted_answer
        else:
            key = row['correct_object'] + " vs. " + row['comparison_object']
            answers[key] = predicted_answer
            
    return answers

# Define a subclass to modify specific layers
class ModifiedMistralDecoderLayer(nn.Module):
    def __init__(self, original_layer, edit_fn, task_vector):
        super().__init__()
        self.original_layer = original_layer
        self.edit_fn = edit_fn  # Function to modify hidden states
        self.task_vector = task_vector 

    def forward(self, hidden_states, *args, **kwargs):
        output = self.original_layer(hidden_states, *args, **kwargs)
        modified_hidden_states = self.edit_fn(output[0], self.task_vector)
        return (modified_hidden_states,) + output[1:]
        
class ModifiedDecoderLayer(nn.Module):
    def __init__(self, original_layer, edit_fn, task_vector):
        super().__init__()
        self.original_layer = original_layer
        self.edit_fn = edit_fn  # Function to modify hidden states
        self.task_vector = task_vector
    # Method to update the task vector after model loading
    def set_task_vector(self, new_task_vector):
        self.task_vector = new_task_vector
        
    def forward(self, hidden_states, *args, **kwargs):
        output = self.original_layer(hidden_states, *args, **kwargs)
        # Assume output[0] contains the hidden states to be modified
        modified_hidden_states = self.edit_fn(output[0], self.task_vector)
        return (modified_hidden_states,) + output[1:]
   

def edit_fn(hidden_states, task_vector):
    """
    Apply the task vector to the last token's hidden state.
    """
    hidden_states[:, -1, :] += task_vector  
    return hidden_states
#"""
def preprocess_answer(answer):
    """
    This function processes the input string by looking for patterns such as:
    - "is larger than"
    - "is usually larger than"
    - "are larger than"
    It extracts the part before the matching phrase, removes "the " if it exists, and strips any extra whitespace.
    """
    answer = answer.replace("the largest object in the image is the ", "")
    # List of patterns to search for
    patterns = [
        r"(.*?)\s+is larger",
        r"(.*?)\s+is usually larger",
        r"(.*?)\s+is generally larger",
        r"(.*?)\s+are usually larger",
        r"(.*?)\s+are generally larger",
        r"(.*?)\s+are larger"
    ]
    
    # Process the input for each pattern
    for pattern in patterns:
        match = re.search(pattern, answer)
        if match:
            answer = match.group(1).replace("the ", "").replace(" in image", "").strip()  # Remove 'the ' and strip whitespace
    return answer
    
def count_flips(task, edit_answers, original_answers, true_answers):
    flip_count = 0
    flip_to_correct_count = 0
    original_not_true = 0
    flip_count_original_not_true = 0
    
    color_mappings = {
        "yellow": ["gold"], 
        "gold": ["orange", "yellow"],
        "purple": ["pink"], 
        "pink": ["purple"],
        "brown": ["orange"], 
        "orange": ["brown"],
        "red": ["red"],
        "black": ["gray", "grey", "silver"],
        "gray": ["black", "silver"],
        "grey": ["black", "silver", "gray"],
        "green": ["green"],
        "blue": ["blue"],
        "silver": ["silver", "gray", "grey"]
    }

    for key in edit_answers:
        
        original = original_answers.get(key) #.lower().replace("_", " ").replace(".", "")
        edit = edit_answers.get(key) #.lower().replace("_", " ").replace(".", "")
        true = true_answers.get(key) #.lower().replace("_", " ").replace(".", "")
        
        """
        if task == "size":

            edit = preprocess_answer(edit)
            original = preprocess_answer(original)
        """
            
        
        if original != edit:
            flip_count += 1

        if original != true:
            
            original_not_true += 1
            
            #if edit != original and edit == true:
       
            # Convert to lowercase to make it case-insensitive
            if task == "color":
            
                if edit != original and (edit.lower() == true or (true in color_mappings and edit in color_mappings[true])):
                        flip_to_correct_count += 1
                        flip_count_original_not_true += 1
                
                elif edit != original:
                    flip_count_original_not_true += 1
            else:
                if edit != original and (edit.lower() == true):
                        flip_to_correct_count += 1
                        flip_count_original_not_true += 1
                
                elif edit != original:
                    flip_count_original_not_true += 1

    print("Flips %: ", flip_count/len(edit_answers))
    print("Eligible Flip to Counterfact %: ", flip_to_correct_count/original_not_true)
    print("Eligible Flip %: ", flip_count_original_not_true/original_not_true)
    
    return original_not_true, flip_count, flip_count_original_not_true, flip_to_correct_count
#"""



def main():     
    parser = argparse.ArgumentParser(description="Run MLLMs on all tasks.")
    parser.add_argument('--model_version', type=str, choices=['llava-next', 'qwen' ,'janus'], required=True, help="Choose the model version.")
    parser.add_argument('--task', type=str, choices=['color', 'size', 'texture'], required=True, help="Choose the task.")
    parser.add_argument('--ablate', default='none', required=True, help="Do you want to ablate the task-vector?") 
    parser.add_argument('--top50', default='false', required=False, help="Ablate top 50 dims?") 
    # To have a list of layers to iterate through. 
    parser.add_argument('--target_layers', required=True, type=int, nargs='+', help="List of target layers as integers.")
    
    args = parser.parse_args()
    
    # Load LLAVA model and processor
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
    text_only = False
    model_name = args.model_version

    if model_name == 'llava-next':

        processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
        model = LlavaNextForConditionalGeneration.from_pretrained(
            "llava-hf/llava-v1.6-mistral-7b-hf", quantization_config=bnb_config, torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        instruction_tokens = "[INST] <image>\n" if not text_only else "[INST]\n"
        end_tokens = "[/INST]"

    elif model_name == 'janus':
        print("loading janus")
        model_path = "deepseek-ai/Janus-Pro-7B"
        processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
        model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
        model = model.to(torch.bfloat16).cuda().eval()
        instruction_tokens = ""
        end_tokens = ""
        instruction_tokens=""
        end_tokens=""

    elif model_name == 'qwen':
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

        model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
        )
        instruction_tokens = ""
        end_tokens = ""
        
    if args.task == "color":
        #df = pd.read_csv("/oscar/data/ceickhof/visual_counterfact/final_images_with_counterfact.csv") #.head(10)
        if model_name == "llava-next":
            df = pd.read_csv("color_filtered_WK.csv") #.head(5)
            
        elif model_name == "qwen":
            df = pd.read_csv("color_qwen_filtered_WK.csv")
            
        elif model_name == "janus":
            df = pd.read_csv("color_janus_filtered_WK.csv") #.head(5)
            
        
        #color_df_for_TV_qwen.csv
        #df_name = f'color_df_for_TV_{args.model_version}.csv'
        #df = pd.read_csv(df_name)

    
        file_path = f"hidden_states_color_{model_name}_575_most_False.pickle"
        with open(file_path, "rb") as f:
            states_counterfact_this = pickle.load(f)
    
        file_path = f"hidden_states_color_{model_name}_575_most_True.pickle"
        with open(file_path, "rb") as f:
            states_counterfact_most = pickle.load(f)
            
    elif args.task == "size":
        #df = pd.read_csv("with_line_sizes_dup.csv") #.head(10)
        
        #df = pd.read_csv("with_line_sizes_balanced.csv")
        
        #df = pd.read_csv("size_filtered_WK.csv")
        
        if model_name == "llava-next":
            df = pd.read_csv("size_df_for_TV_llava-next.csv") #.head(5)
            
        elif model_name == "qwen":
            df = pd.read_csv("size_df_for_TV_qwen.csv")
            
        elif model_name == "janus":
            df = pd.read_csv("size_df_for_TV_janus.csv") #.head(5)
            
    
        file_path = f"hidden_states_size_{model_name}_1754_most_False.pickle"
        #file_path = "hidden_states_size_llava-next_877_most_False.pickle"
        with open(file_path, "rb") as f:
            states_counterfact_this = pickle.load(f)
    
        file_path = f"hidden_states_size_{model_name}_1754_most_True.pickle"
        #file_path = "hidden_states_size_llava-next_877_most_True.pickle"
        with open(file_path, "rb") as f:
            states_counterfact_most = pickle.load(f)
        

    image_type = 'counterfact'
    most = 'True' #'True' #FALSE
    original_answers = get_answers(model, model_name, args.task, processor, df, image_type, most, instruction_tokens, end_tokens)

    # SPECIFY THE TARGET LAYERS TO ADD THE TASK VECTOR
    TARGET_LAYERS = args.target_layers 
    print(TARGET_LAYERS)

    # outliers
    outlier_indices_tensor = torch.tensor([2070, 2078, 3901], device=model.device)    
    
    if args.ablate == 'keep-k':
        outlier_indices_tensor = {
        layer_idx: torch.topk(
            torch.tensor( # SWAP HERE
                np.abs(np.mean(states_counterfact_this[layer_idx] - states_counterfact_most[layer_idx], axis=0))
                #np.abs(np.mean(states_counterfact_most[layer_idx] - states_counterfact_this[layer_idx], axis=0))
            ),
            100  # Select the top 80 dimensions
        ).indices.to(model.device)
        for layer_idx in TARGET_LAYERS
    }

        task_vectors = {
            layer_idx: torch.zeros(4096, dtype=torch.float16, device=model.device)
            .index_put_((outlier_indices_tensor[layer_idx],), torch.tensor( #SWAP HERE
                np.mean(states_counterfact_this[layer_idx] - states_counterfact_most[layer_idx], axis=0),
                #np.mean(states_counterfact_most[layer_idx] - states_counterfact_this[layer_idx], axis=0),
                dtype=torch.float16,
                device=model.device
            )[outlier_indices_tensor[layer_idx]]).to(model.device)
            for layer_idx in TARGET_LAYERS
        }
    
    if args.ablate == 'none':
        task_vectors = {
            layer_idx: torch.tensor(
                # computing the task-vector at layer_idx #SWAP HERE
                np.mean(states_counterfact_this[layer_idx] - states_counterfact_most[layer_idx], axis=0), 
                #np.mean(states_counterfact_most[layer_idx] - states_counterfact_this[layer_idx], axis=0),
                dtype=torch.float16
            ).to(model.device)
            for layer_idx in TARGET_LAYERS
        }

    if args.ablate == 'non-outliers':
        # ablate all non-outliers. 
        # only add outliers in tv
        task_vectors = {
            layer_idx: torch.zeros(4096, dtype=torch.float16, device=model.device)  # Initialize a zero tensor
            .index_put_((outlier_indices_tensor,), torch.tensor( #SWAP HERE
                np.mean(states_counterfact_this[layer_idx] - states_counterfact_most[layer_idx], axis=0), 
                #np.mean(states_counterfact_most[layer_idx] - states_counterfact_this[layer_idx], axis=0), 
                dtype=torch.float16,
                device=model.device  # Ensure the tensor is created on the correct device
            )[outlier_indices_tensor]).to(model.device)  # Set values at specific indices
            for layer_idx in TARGET_LAYERS
        }

    if args.ablate == 'only-outliers': 
        # Compute the mean differences for each target layer.
        task_vectors = {
            layer_idx: torch.tensor( #SWAP HERE
                np.mean(states_counterfact_this[layer_idx] - states_counterfact_most[layer_idx], axis=0),
                #np.mean(states_counterfact_most[layer_idx] - states_counterfact_this[layer_idx], axis=0),
                dtype=torch.float16,
                device=model.device
            )
            for layer_idx in TARGET_LAYERS
        }

        for layer_idx in TARGET_LAYERS:
            # Use in-place assignment to set specified indices to 0.
            task_vectors[layer_idx][outlier_indices_tensor.to(torch.int64)] = 0.0

    # Replace the target layers in the Mistral model
    if model_name == "llava-next":
        for layer_idx, task_vector in task_vectors.items():
            original_layer = model.language_model.model.layers[layer_idx]
            model.language_model.model.layers[layer_idx] = ModifiedMistralDecoderLayer(
                original_layer, edit_fn, task_vector
            )
    elif model_name == "qwen":
        for layer_idx, task_vector in task_vectors.items():
            original_layer = model.model.layers[layer_idx]
            model.model.layers[layer_idx] = ModifiedDecoderLayer(
                original_layer, edit_fn, task_vector
            )
    elif model_name == "janus":
        for layer_idx, task_vector in task_vectors.items():
            original_layer = model.language_model.model.layers[layer_idx]
            model.language_model.model.layers[layer_idx] = ModifiedMistralDecoderLayer(
                original_layer, edit_fn, task_vector
            )

    edit_answers = get_answers(model, model_name, args.task, processor, df, image_type, most, instruction_tokens, end_tokens)


    true_answers = {}
    for idx, row in df.iterrows(): 
        if args.task == "color":
            true_answers[row['correct_object']] = row['incorrect_answer'] #row[f'{args.model_version}_lm_preds'].lower().replace("_", " ").replace(".", "") #row['incorrect_answer'] #correct_answer 
        else:
            key = row['correct_object'] + " vs. " + row['comparison_object']
            true_answers[key] = row['incorrect_answer']

    
    for key in edit_answers:
        # Clean the original, edited, and true answers (lowercase, replace underscores, remove periods) in place
        original_answers[key] = original_answers.get(key).lower().replace("_", " ").replace(".", "")
        edit_answers[key] = edit_answers.get(key).lower().replace("_", " ").replace(".", "")
        true_answers[key] = true_answers.get(key).lower().replace("_", " ").replace(".", "")
        
        # If task is "size", preprocess the answers in place
        if args.task == "size":
            edit_answers[key] = preprocess_answer(edit_answers[key])  # Update in-place
            original_answers[key] = preprocess_answer(original_answers[key])
            
    print("OG: ", original_answers)
    print("TRUE: ", true_answers)
    print("EDIT: ", edit_answers)

    count_flips(args.task, edit_answers, original_answers, true_answers)

if __name__ == '__main__':
    main()



    

    


