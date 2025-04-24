import torch
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from clinicalt5_moe import ClinicalT5MoE
from ClinicalNoteDataset import ClinicalNoteDataset
from utils import predict_icd_codes, save_model


def train_clinical_t5_moe(
        model, 
        train_dataloader, 
        valid_dataloader, 
        num_epochs=3
        ):

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(num_epochs):

        # Train
        model.train()
        total_loss = 0
        
        for batch in train_dataloader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in valid_dataloader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
                
        avg_val_loss = val_loss / len(valid_dataloader)
        print(f"Epoch {epoch}, Valid Loss: {avg_val_loss:.4f}")
    
    return model


def main():

    clinicalt5_moe = ClinicalT5MoE(
        model_name="luqh/ClinicalT5-base",  # Start with standard T5
        num_experts=16,
        use_hierarchical=False
    )
    
    # Example
    clinical_notes = ["Patient presents with...", "History of..."]
    icd_codes = ["ICD-10:J18.9,E11.9", "ICD-10:I10,E78.5"]

    # Actual Data:
    
    
    # Dataloader
    dataset = ClinicalNoteDataset(
        clinical_notes=clinical_notes,
        icd_codes=icd_codes,
        tokenizer=clinicalt5_moe.tokenizer
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    valid_dataloader = DataLoader(val_dataset, batch_size=2)
    
    # Train
    model = train_clinical_t5_moe(
        model=clinicalt5_moe,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,  # Use separate validation data in practice
        num_epochs=3
    )

    # Example
    sample_note = "Patient is tired easily, bloated, and having trouble pooping. No fiber intake in the past month."
    predicted_codes = predict_icd_codes(clinical_t5_moe, sample_note, clinical_t5_moe.tokenizer)
    print(f"Predicted ICD Codes: {predicted_codes}")

    # Save model
    save_model(model, 'clinicalt5_moe')


if __name__ == "__main__":
    main()
