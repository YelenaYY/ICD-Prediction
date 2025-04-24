import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional
from transformers import T5Tokenizer


# Dummy for testing purpose
# class ClinicalNoteDataset(Dataset):

#     def __init__(self, clinical_notes, icd_codes, tokenizer, max_length=512):

#         self.tokenizer = tokenizer
#         self.clinical_notes = clinical_notes
#         self.icd_codes = icd_codes
#         self.max_length = max_length
        
#     def __len__(self):
#         return len(self.clinical_notes)
        
#     def __getitem__(self, idx):

#         note = self.clinical_notes[idx]
#         codes = self.icd_codes[idx]
        
#         input_text = f"Clinical note: {note} Predict ICD codes: <extra_id_0>"
#         target_text = f"<extra_id_0> {codes} <extra_id_1>"
        
#         inputs = self.tokenizer(
#             input_text, 
#             max_length=self.max_length,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt"
#         )
        
#         targets = self.tokenizer(
#             target_text,
#             max_length=128,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt"
#         )
        
#         data = {
#             "input_ids": inputs.input_ids.squeeze(),
#             "attention_mask": inputs.attention_mask.squeeze(),
#             "labels": targets.input_ids.squeeze()
#         }

#         return data



class ClinicalNoteDataset(Dataset):
    """Dataset for clinical notes and ICD codes from MIMIC-IV."""
    
    def __init__(
        self,
        clinical_notes: list[str],
        tokenizer: T5Tokenizer,
        max_length: int = 512,
        split: str = 'train',
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize the dataset.
        
        Args:
            discharge_path: Path to discharge notes CSV
            diagnoses_path: Path to diagnoses ICD codes CSV
            tokenizer: T5Tokenizer instance
            max_length: Maximum sequence length
            split: One of 'train', 'val', or 'test'
            test_size: Proportion of data to use for testing
            val_size: Proportion of training data to use for validation
            random_state: Random seed for reproducibility
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load and process data
        discharge_df = pd.read_csv(discharge_path, compression='gzip')
        diagnoses_df = pd.read_csv(diagnoses_path, compression='gzip')
        
        # Merge data on hadm_id
        merged_df = pd.merge(
            discharge_df[['hadm_id', 'text']],
            diagnoses_df[['hadm_id', 'icd_code', 'icd_version']],
            on='hadm_id',
            how='inner'
        )
        
        # Group ICD codes by hadm_id
        icd_groups = merged_df.groupby('hadm_id').agg({
            'text': 'first',
            'icd_code': lambda x: list(x)
        }).reset_index()
        
        # Split data
        from sklearn.model_selection import train_test_split
        train_val_df, test_df = train_test_split(
            icd_groups,
            test_size=test_size,
            random_state=random_state
        )
        
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size,
            random_state=random_state
        )
        
        # Select appropriate split
        if split == 'train':
            self.data = self._prepare_data(train_df)
        elif split == 'val':
            self.data = self._prepare_data(val_df)
        elif split == 'test':
            self.data = self._prepare_data(test_df)
        else:
            raise ValueError(f"Invalid split: {split}. Must be one of 'train', 'val', or 'test'")
    
    def _prepare_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Prepare data in the required format."""
        data = []
        for _, row in df.iterrows():
            data.append({
                'text': row['text'],
                'labels': row['icd_code']
            })
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset."""
        item = self.data[idx]
        encoding = self.tokenizer(
            item['text'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': item['labels']
        }
