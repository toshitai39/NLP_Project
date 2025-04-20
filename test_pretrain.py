from unsloth import FastLanguageModel
import torch
import pandas as pd
from tqdm import tqdm

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "./outputs_instruction_tuned_only/checkpoint-50", # or choose "unsloth/Llama-3.2-1B-Instruct"
    # model_name="ArkaAcharya/LLAMA_IITP_1B_PRETRAIN",
    # model_name="./outputs_instruction_tuned/checkpoint-250",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)
# print(model)
# model=model.merge_and_unload()
# print(model)
chat_prompt = """
### Instruction:
{}

### Input:
{}

### Response:
{}"""

test_data=pd.read_csv('final_qa_dataset_test.csv')
generated_outputs=[]
for i in tqdm(range(len(test_data))):
    
    text=chat_prompt.format(
            "", # instruction - leave this blank!
            f"{test_data['Question'][i]}", # input
            "", # output - leave this blank!
        )
    #inputs=tokenizer(text,return_tensors='pt')
    FastLanguageModel.for_inference(model)
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True).to("cuda")
    
    outputs = model.generate(**inputs, max_new_tokens=250, num_return_sequences=1)
    
    text_out = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    generated_outputs.append(text_out)

test_data['Generated'] = generated_outputs
test_data.to_csv('generated_outputs_test_baseline_only_pt.csv',index=False)

# from transformers import AutoTokenizer
# from unsloth import FastLanguageModel

# # Load tokenizer first

# # Load model with tokenizer override
# base_model, _ = FastLanguageModel.from_pretrained(
#     model_name = "unsloth/Llama-3.2-1B-Instruct",
#     max_seq_length = max_seq_length,
#     dtype = dtype,
#     load_in_4bit = load_in_4bit,
# )


# # Attach saved LoRA adapter


# from unsloth import FastLanguageModel
# import torch
# max_seq_length = 1024 # Choose any! We auto support RoPE Scaling internally!
# dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
# load_in_4bit = False # Use 4bit quantization to reduce memory usage. Can be False.

# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = "unsloth/Llama-3.2-1B-Instruct", # Choose ANY! eg teknium/OpenHermes-2.5-Mistral-7B
#     max_seq_length = max_seq_length,
#     dtype = dtype,
#     load_in_4bit = load_in_4bit,
#     fix_tokenizer=False
#     # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
# )
# model.load_adapter("ArkaAcharya/LLAMA_IITP_1B_PRETRAIN")
# model.save_pretrained_merged(
#     "./merged_LLAMA_IITP_1B_new",
#     tokenizer,
#     save_method = "merged_16bit",  # Preserves precision
#     push_to_hub = False,
# )