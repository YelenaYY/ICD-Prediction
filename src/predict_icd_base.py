import torch
from clinicalt5_base import BaseT5Model
from tqdm import tqdm

def predict_icd_codes(
    clinical_notes: list,
    model_folder_path: str = "luqh/ClinicalT5-base",
    embedding_file: list[str] = None,
    batch_size: int = 8
):
    """
    Use BaseT5Model to predict ICD codes from clinical notes.
    
    Args:
        clinical_notes: List of clinical notes
        model_folder_path: Path to the pretrained ClinicalT5 model
        embedding_file: List of embedding files (optional)
        batch_size: Batch size for prediction
        
    Returns:
        List of predicted ICD codes
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model with Flax weights
    model = BaseT5Model(model_folder_path=model_folder_path, embedding_file=embedding_file, from_flax=True)
    
    # Move model to device and set to eval mode
    model.model = model.model.to(device)
    model.model.eval()
    
    # Process notes in batches
    predicted_codes = []
    for i in tqdm(range(0, len(clinical_notes), batch_size)):
        batch_notes = clinical_notes[i:i + batch_size]
        
        # Get predictions for each note in the batch
        batch_predictions = []
        for note in batch_notes:
            with torch.no_grad():
                codes = model.run_t5model(note)
                batch_predictions.append(codes)
        
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
        model_folder_path="luqh/ClinicalT5-base"
    )
    
    # Print results
    for note, codes in zip(clinical_notes, predicted_codes):
        print(f"\nClinical Note: {note}")
        print(f"Predicted ICD Codes: {codes}")

if __name__ == "__main__":
    main() 