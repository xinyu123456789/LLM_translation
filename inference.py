import torch
import csv
import re
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("MediaTek-Research/Breeze-7B-32k-Instruct-v1_0")
model = AutoModelForCausalLM.from_pretrained(
   "MediaTek-Research/Breeze-7B-32k-Instruct-v1_0",
   device_map="auto",
   torch_dtype=torch.float16,

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
def is_safe_for_word_boundary(term):
    """
    判斷術語是否只包含適用於 word boundary 的字元（字母、數字、底線、連字號）
    """
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


term_dict = "terms2.csv"
text = "In modern software development, integrating third-party RESTful APIs has become essential for enhancing functionality without reinventing the wheel. Developers often rely on API endpoints provided by cloud services like Google Maps or Stripe to perform tasks such as payment processing or geolocation. When accessing these APIs, an API key is typically required for authentication and to monitor usage quotas. To ensure security and compatibility, many teams use OpenAPI specifications to standardize how their APIs are documented and consumed. Additionally, implementing rate limiting on public-facing APIs is critical to prevent abuse and maintain server performance. Whether you're building a microservices architecture or a monolithic application, understanding how to interact with and document APIs is fundamental for scalable and maintainable software design."

# 替換專有名詞
new_text = replace_terms(text, term_dict)
print(new_text)

system_prompt = {
    "role": "system", "content": "你現在是一個專精於科技領域英文到繁體中文的翻譯人員。"
}

chat_text = "請將下列英文句子翻譯成繁體中文，但不要翻譯被 <ter>...</ter> 標籤包住的術語，這些術語和標籤請原封不動保留，英文句子: " + new_text

chat = [
  system_prompt,
  {"role": "user", "content": chat_text},   #提問
]
prompt = tokenizer.apply_chat_template(chat, tokenize=False)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=2048,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n=== 模型回應 ===\n")
print(response)
