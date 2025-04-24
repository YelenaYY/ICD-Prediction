import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
from tqdm import tqdm

def predict_icd_codes(
    clinical_notes: list,
    model_name: str = "luqh/ClinicalT5-base",
    max_length: int = 512,
    batch_size: int = 8
):
    """
    Use pretrained ClinicalT5 model to predict ICD codes from clinical notes.
    
    Args:
        clinical_notes: List of clinical notes
        model_name: Name of the pretrained ClinicalT5 model
        max_length: Maximum sequence length
        batch_size: Batch size for prediction
        
    Returns:
        List of predicted ICD codes
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load pretrained tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with proper weight assignment
    model = T5ForConditionalGeneration.from_pretrained(
        model_name,
        from_flax=True
        # torch_dtype=torch.float32,
        # low_cpu_mem_usage=True
    )
    
    # Initialize model on device
    model = model.to_empty(device=device)
    model.eval()
    
    # Process notes in batches
    predicted_codes = []
    for i in tqdm(range(0, len(clinical_notes), batch_size)):
        batch_notes = clinical_notes[i:i + batch_size]
        
        # Tokenize input
        inputs = tokenizer(
            batch_notes,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate predictions with specific parameters
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=100,
                num_beams=4,
                early_stopping=True,
                do_sample=False,
                temperature=1.0,
                top_k=50,
                top_p=1.0,
                repetition_penalty=1.0
            )
        
        # Decode predictions
        batch_predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predicted_codes.extend(batch_predictions)
    
    return predicted_codes

def main():
    # Example usage
    clinical_notes = [
        "Patient presents with fever and cough. History of diabetes.",
        "Patient has chest pain and shortness of breath. No prior cardiac history."
    ]
    
    # Get predictions
    predicted_codes = predict_icd_codes(
        clinical_notes=clinical_notes,
        model_name="luqh/ClinicalT5-base"
    )
    
    # Print results
    for note, codes in zip(clinical_notes, predicted_codes):
        print(f"\nClinical Note: {note}")
        print(f"Predicted ICD Codes: {codes}")

if __name__ == "__main__":
    main() 