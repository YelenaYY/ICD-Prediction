import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer

class MIMICDataset(Dataset):
    """Dataset for MIMIC-IV discharge notes and ICD codes."""
    
    def __init__(
        self,
        discharge_path: str,
        diagnoses_path: str,
        tokenizer: T5Tokenizer,
        max_length: int = 512,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ):
        """
        Initialize the MIMIC dataset.
        
        Args:
            discharge_path: Path to discharge notes CSV
            diagnoses_path: Path to diagnoses ICD codes CSV
            tokenizer: T5Tokenizer instance
            max_length: Maximum sequence length
            test_size: Proportion of data to use for testing
            val_size: Proportion of training data to use for validation
            random_state: Random seed for reproducibility
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load and process data
        self.discharge_df = pd.read_csv(discharge_path, compression='gzip')
        self.diagnoses_df = pd.read_csv(diagnoses_path, compression='gzip')
        
        # Merge data on hadm_id
        self.merged_df = pd.merge(
            self.discharge_df[['hadm_id', 'text']],
            self.diagnoses_df[['hadm_id', 'icd_code', 'icd_version']],
            on='hadm_id',
            how='inner'
        )
        
        # Group ICD codes by hadm_id
        self.icd_groups = self.merged_df.groupby('hadm_id').agg({
            'text': 'first',
            'icd_code': lambda x: list(x)
        }).reset_index()
        
        # Split data
        train_val_df, test_df = train_test_split(
            self.icd_groups,
            test_size=test_size,
            random_state=random_state
        )
        
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size,
            random_state=random_state
        )
        
        self.train_data = self._prepare_data(train_df)
        self.val_data = self._prepare_data(val_df)
        self.test_data = self._prepare_data(test_df)
    
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
        return len(self.train_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset."""
        item = self.train_data[idx]
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
    
    def get_dataloaders(
        self,
        batch_size: int = 8,
        num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get training, validation, and test dataloaders.
        
        Args:
            batch_size: Batch size for dataloaders
            num_workers: Number of workers for data loading
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        train_loader = DataLoader(
            self.train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            self.val_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            self.test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader

def create_mimic_dataloaders(
    self,
    discharge_path: str,
    diagnoses_path: str,
    tokenizer: T5Tokenizer,
    batch_size: int = 8,
    max_length: int = 512,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create dataloaders for MIMIC-IV data.
    
    Args:
        discharge_path: Path to discharge notes CSV
        diagnoses_path: Path to diagnoses ICD codes CSV
        tokenizer: T5Tokenizer instance
        batch_size: Batch size for dataloaders
        max_length: Maximum sequence length
        test_size: Proportion of data to use for testing
        val_size: Proportion of training data to use for validation
        random_state: Random seed for reproducibility
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    dataset = MIMICDataset(
        discharge_path=discharge_path,
        diagnoses_path=diagnoses_path,
        tokenizer=tokenizer,
        max_length=max_length,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state
    )
    
    return dataset.get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers
    ) 
