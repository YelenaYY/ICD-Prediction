import re
import torch

def predict_icd_codes(model, text, tokenizer, max_length=512):
    
    model.eval()
    device = next(model.parameters()).device
    
    input_text = f"Clinical Note: {text} Predict ICD codes: <extra_id_0>"
    
    # Tokenize
    inputs = tokenizer(
        input_text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    # Generate prediction
    with torch.no_grad():
        outputs = model.model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
    
    # Decode
    predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract ICD codes
    codes = re.search("<extra_id_0>(.*?)<extra_id_1>", predicted_text)
    if codes:
        return codes.group(1).strip()
    return "No ICD codes predicted"


def save_model(model, name):
    model.model.save_pretrained("clinicalt5_moe")
    model.tokenizer.save_pretrained("clinicalt5_moe")