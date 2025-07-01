import torch
import csv
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, BitsAndBytesConfig

print("GPU 可用:", torch.cuda.is_available())
print("可用 GPU 數量:", torch.cuda.device_count())
print("使用中的 GPU 名稱:", torch.cuda.get_device_name(0))

model_path = "mistral-translation-train\checkpoint-100"
print(type(model_path))
# 要翻譯的句子
text = "7. 放射線報告(Radiological Report):\\nClinical Information: 1-day history of right upper quadrant abdominal pain radiating to the back, associated with nausea and vomiting; suspected 結石性急性膽囊炎.\\n<<Main Findings>>\\n1. Gallbladder wall thickening measuring approximately 5 mm (正常 ≤ 3 mm)\\n2. Multiple gallstones with posterior acoustic shadowing\\n3. Pericholecystic fluid collection (~10–15 mL)\\n4. 超音波墨菲氏徵象陽性\\n<<OtherFindings>>\\n1. Liver: normal size, shape, and echotexture; no focal lesions; intrahepatic bile ducts not dilated\\n2. Common bile duct: 4 mm diameter (正常 ≤ 6 mm), no evidence of choledocholithiasis\\n3. Pancreas: normal morphology; main pancreatic duct not dilated\\n4. Kidneys and spleen: normal bilaterally; no hydronephrosis or splenomegaly\\n5. No free intraperitoneal or pleural fluid\\n Impression:\\nUltrasound findings are consistent with 結石性急性膽囊炎."
# 對照表
term_dict = "terms2.csv"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
   model_path,
   device_map="auto",
   offload_folder="./offload_dir",
   torch_dtype=torch.float16,
   quantization_config=bnb_config
   
)

"""
def load_term_dict(csv_path):
  # 讀取對照表
    term_dict = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            en = row["en"].strip().lower()
            zh = row["zh"].strip()
            term_dict[en] = zh
    return term_dict
"""
"""
def is_safe_for_word_boundary(term):

    #判斷術語是否只包含適用於 word boundary 的字元（字母、數字、底線、連字號）
    return re.match(r'^[\w\-]+$', term) is not None

def replace_terms(text, csv_path):
    # 保留已標記過的範圍（避免嵌套）
    marked_ranges = []

    def is_inside_marked(start, end):
        for s, e in marked_ranges:
            if start < e and end > s:
                return True
        return False

    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            en_term = row['en'].strip()
            if not en_term:
                continue

            safe_en_term = re.escape(en_term)
            if is_safe_for_word_boundary(en_term):
                pattern = re.compile(rf'(?<!\w)({safe_en_term})(?!\w)', re.IGNORECASE)
            else:
                pattern = re.compile(f'({safe_en_term})', re.IGNORECASE)

            def add_tag(match):
                start, end = match.span()
                if is_inside_marked(start, end) or re.search(r'<\/?ter>', match.group(0), re.IGNORECASE):
                    return match.group(0)
                replacement = f"<ter>{match.group(0)}</ter>"
                # 更新區間避免重複包裹
                marked_ranges.append((start, start + len(replacement)))
                return replacement

            text = pattern.sub(add_tag, text)

    return text

# 替換專有名詞
#new_text = replace_terms(text, term_dict)
#print(new_text)
"""
generation_config = GenerationConfig(
    max_new_tokens=5000,
    do_sample=False,
    temperature=0.01,
    top_p=0.01,
    repetition_penalty=1.1,
    eos_token_id = tokenizer.eos_token_id
)

chat = [
  {"role": "user", "content": text},   #提問
]

def _inference(tokenizer, model, generation_config, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output_tensors = model.generate(**inputs, generation_config=generation_config)
    output_str = tokenizer.decode(output_tensors[0])
    return output_str

prompt = tokenizer.apply_chat_template(chat, tokenize=False)
output_str = _inference(tokenizer, model, generation_config, prompt)
print("\n=== 模型回應 ===\n")
print(output_str)
