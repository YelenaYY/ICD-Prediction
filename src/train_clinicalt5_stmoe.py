import re
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import torch.cuda.amp as amp
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader
from clinicalt5_stmoe import ClinicalT5STMoE
from ClinicalNoteDataset import ClinicalNoteDataset
from utils import predict_icd_codes, save_model


def train_clinical_t5_moe(
        model, 
        train_dataloader, 
        valid_dataloader, 
        num_epochs=3,
        learning_rate=5e-5, 
        warmup_steps=1000, 
        gradient_accumulation_steps=4
        ):
    
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    scaler = torch.amp.GradScaler()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Train
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            with torch.amp.autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (step + 1) % gradient_accumulation_steps == 0:
                # Unscale gradients for clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * gradient_accumulation_steps
            
            if (step + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Step {step+1}/{len(train_dataloader)} | Loss: {loss.item() * gradient_accumulation_steps:.4f}")
            
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch}/{num_epochs}, Average Train loss: {avg_train_loss:.4f}")
        
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
        print(f"Epoch {epoch}/{num_epochs}, Validation loss: {avg_val_loss:.4f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_train_loss,
        }, f"clinical_t5_moe_epoch_{epoch}.pt")
    
    return model


def main():
    
    clinical_t5_moe = ClinicalT5STMoE(
        model_name="luqh/ClinicalT5-base",
        num_experts=16,
        expert_hidden_mult=4,
        threshold_train=0.2,
        threshold_eval=0.2,
        gating_top_n=2,
        add_ff_before=False,
        add_ff_after=True
    )
    
    # Example
    clinical_notes = ["Patient presents with...", "History of..."]
    icd_codes = ["ICD-10:J18.9,E11.9", "ICD-10:I10,E78.5"]
    
    # Dataloader
    dataset = ClinicalNoteDataset(
        clinical_notes=clinical_notes,
        icd_codes=icd_codes,
        tokenizer=clinical_t5_moe.tokenizer
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    valid_dataloader = DataLoader(val_dataset, batch_size=2)
    
    # Train
    model = train_clinical_t5_moe(
        model=clinical_t5_moe,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
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
