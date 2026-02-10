from transformers import AutoModel, AutoTokenizer
import torch
from PIL import Image
import os

model_name = 'deepseek-ai/DeepSeek-OCR'
print(f"Loading OCR model: {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.bfloat16)
model = model.eval().cuda()

image_path = '/home/user/workspace/vllm/Screenshot 2025-10-08 at 9.17.49 PM.png'
output_path = '/home/user/lab_env/ml-explorations/2026-02-01_deepseek-r1-distill-llama-70b-vllm/ocr_output'

# DeepSeek-OCR specific infer method
# prompt = "<image>\nExtract all text from this image."
prompt = "<image>\nParse the figure. "

print("Running OCR via .infer()...")
with torch.no_grad():
    # The .infer method handles image loading, preprocessing, and generation
    response = model.infer(
        tokenizer, 
        prompt=prompt, 
        image_file=image_path, 
        output_path=output_path,
        save_results=True
    )
    print("\n" + "="*20)
    print(f"OCR Result: {response}")
    print("="*20)
