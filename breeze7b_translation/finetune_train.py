"""
    用來對 LLM 模型進行微調(只接受純文本)
    使用情境:
        單張顯卡且 vram 超級小(不超過 8 GB)
"""
import re
from accelerate import Accelerator
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from peft import (
    LoraConfig,
    #PeftConfig,
    #PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
import time
import gc
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import transformers
#from datetime import datetime

print("GPU 可用:", torch.cuda.is_available())
print("可用 GPU 數量:", torch.cuda.device_count())
print("使用中的 GPU 名稱:", torch.cuda.get_device_name(0))

# 調整 CUDA 記憶體使用策略
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True, max_split_size_mb:128"
min_memory_available = 2 * 1024 * 1024 * 1024  # 2GB，要丟進 GPU 前最小的必須空間

# 開始前先清空 GPU 的記憶體
def clear_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()

# 先確定 GPU 有足夠的記憶體空間
def wait_until_enough_gpu_memory(min_memory_available, max_retries=10, sleep_time=5):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

    for _ in range(max_retries):
        info = nvmlDeviceGetMemoryInfo(handle)
        if info.free >= min_memory_available:
            break
        print(f"Waiting for {min_memory_available} bytes of free GPU memory. Retrying in {sleep_time} seconds...")
        time.sleep(sleep_time)
    else:
        raise RuntimeError(f"Failed to acquire {min_memory_available} bytes of free GPU memory after {max_retries} retries.")

clear_gpu_memory()
#wait_until_enough_gpu_memory(min_memory_available)

# 訓練集和驗證集要用 json 格式
train_dataset = load_dataset('json', data_files='medical_sentences_train.jsonl', split='train') # data_files 填檔名，記得副檔名要用 jsonl
eval_dataset = load_dataset('json', data_files='medical_sentences_eval.jsonl', split='train') # data_files 填檔名，記得副檔名要用 jsonl

accelerator = Accelerator(mixed_precision="fp16")   # 啟用混合精度訓練

base_model_id = "MediaTek-Research/Breeze-7B-32k-Instruct-v1_0" # 準備被微調的模型

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True, # 啟用雙重量化，節省 VRAM
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)
# 導入要訓練的模型
model = AutoModelForCausalLM.from_pretrained(
    base_model_id, 
    quantization_config=bnb_config, 
    device_map = 'auto',
    offload_folder = 'offload_dir',
    offload_state_dict = True,
    torch_dtype = torch.float16,
    max_memory = {
        0: "5GB",
        "cpu": "15GB"
    }
    )
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)  # 配合 4 bit 量化

def formatting_func(data):
    # 整理資料的格式
    text = f"{data['input']}\n\n{data['output']}"
    return text

#導入 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",    # 設定填補的 <pad> 放在句子左邊，如果句子長度不夠
    add_eos_token=True,     # 輸入和輸出都自動加入 <eos> end of sentence
    add_bos_token=True,     # 輸入前都自動加上 <bos> beginning of sentence
)
tokenizer.pad_token = tokenizer.eos_token   # 把所有 <pad> 改成 <eos>

def generate_and_tokenize_prompt(prompt):       # 處理後的 dataset 可以用來當驗證集
    # 用來處理驗證集的格式(沒有 label)
    return tokenizer(formatting_func(prompt))

# 這邊先處理一次資料集(測試用，正式訓練前可以刪掉)
tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)   # 如果最後要正式訓練前真的學的多於就刪掉
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)      # 整理驗證集

max_length = 3000 # 最長的 token

def generate_and_tokenize_prompt2(prompt):
    # 用來處理訓練集的格式
    result = tokenizer(
        formatting_func(prompt),
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()

    return result

# 創建一個函數來獲取合法的參數名稱(有些參數名稱會有 .(會被當作有意義) ，很煩)
def get_legal_param_name(name):
    return re.sub(r"\.", "_", name)


def convert_model_to_float16(model):
    # 用來處理不是 float16 的參數和違法的參數名稱
    for param_name, param in model.named_parameters():
        legal_name = get_legal_param_name(param_name)
        
        # 在迴圈內創建新的Parameter對象
        param_value_f16 = torch.nn.Parameter(param.data.to(torch.float16))
        
        # 用新創建的float16 Parameter對象替換原始參數
        setattr(model, legal_name, param_value_f16)
        print(f"Parameter {legal_name}, converted dtype: {param_value_f16.dtype}")
        
    for buffer_name, buffer_value in model.named_buffers():
        # 和上面那個 for 差不多作用，只是處理的對象不一樣
        legal_name = get_legal_param_name(buffer_name)
        buffer_value_f16 = buffer_value.to(torch.float16)
        model.register_buffer(legal_name, buffer_value_f16)
        print(f"Buffer {legal_name}, converted dtype: {buffer_value_f16.dtype}")
        
# 遍歷模型命名參數並打印原始數據類型  檢查有沒有用成 float16
for name, param in model.named_parameters():
    print(f"Parameter {name}, original dtype: {param.dtype}")
    
for name, buffer in model.named_buffers():
    print(f"Buffer {name}, original dtype: {buffer.dtype}")
    
# 調用函數將模型參數和buffer類型轉換為float16
#convert_model_to_float16(model)  # 這行用來補救，如果上面print 出來還有 float32 就加這行進去

clear_gpu_memory()
#wait_until_enough_gpu_memory(min_memory_available)

# 這邊才是正式訓練前的 dataset 轉換格式
tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt2)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)

def print_trainable_parameters(model):      #確認有多少可用來訓練的參數(如果不想看到可以刪掉)
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

# 終於要準備開始訓練了
# 做 QLoRA 的 Config
config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,  # Conventional
    task_type="CAUSAL_LM",     # 任務類型
)

clear_gpu_memory()
model = get_peft_model(model, config)   # 啟用 LoRA 微調機制
model.config.gradient_checkpointing = True  # 開啟梯度檢查點
print_trainable_parameters(model)   # 不爽的話可以刪掉

model = accelerator.prepare_model(model)


project = "translation-train"    # 微調專案的名稱
base_model_name = "mistral"     # 微調的模型
run_name = base_model_name + "-" + project  # 組合成這次訓練的名字
output_dir = "./" + run_name    # 用來當作這次訓練的資料夾名稱

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args = transformers.TrainingArguments(
        output_dir = output_dir,
        warmup_steps = 1,
        dataloader_pin_memory = False,
        per_device_train_batch_size = 1,
        per_device_eval_batch_size = 1,
        gradient_accumulation_steps = 2,
        gradient_checkpointing = True,
        max_steps = 100,
        learning_rate = 2e-5,
        bf16 = False,
        fp16 = True,
        optim = "paged_adamw_8bit",
        logging_steps = 1,
        logging_dir = "./logs",
        save_strategy = "steps",
        save_steps = 20,
        eval_strategy = "steps",        # 原版是 evaluation_strategy，但改了
        eval_steps = 10,
        do_eval = True
        #report_to="wandb",           # 看想不想用 wandb 想用再用
        #run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",          # Name of the W&B run (optional)(要用 wandb 再用就好)
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# 這邊直接開始練爆
trainer.train()