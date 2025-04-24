# ICD Code Prediction

This project implements an automated ICD (International Classification of Diseases) code prediction system using Clinical T5 models. The system processes clinical notes and predicts relevant ICD-10 codes.

## Project Structure

```
ICD-Prediction/
├── src/
│   ├── clinicalt5_base.py      # Base T5 model implementation
│   ├── clinicalt5_moe.py       # Mixture of Experts T5 implementation
│   ├── clinicalt5_stmoe.py     # Sparse/Switch Transformer MoE implementation
│   ├── predict_icd_base.py     # Prediction script using base model
│   ├── predict_icd.py          # General prediction utilities
│   ├── ClinicalNoteDataset.py  # Dataset class for clinical notes
│   ├── mimic_data_loader.py    # MIMIC-IV data loading utilities
│   ├── train_clinicalt5_moe.py # Training script for MoE model
│   └── train_clinicalt5_stmoe.py # Training script for ST-MoE model
├── data/                       # Directory for MIMIC-IV data (not included)
└── environment.yml            # Conda environment specification
```

## Setup

1. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate icd-prediction
```

2. Download the MIMIC-IV dataset and place the following files in the `data/` directory:
   - `discharge.csv.gz`: Contains clinical discharge notes
   - `diagnoses_icd.csv.gz`: Contains ICD code annotations

## Models

The project implements three variants of T5-based models:

1. **Base Clinical T5**: Standard T5 model fine-tuned on clinical text
2. **MoE Clinical T5**: T5 with Mixture of Experts layers
3. **ST-MoE Clinical T5**: T5 with Sparse/Switch Transformer MoE architecture

## Usage

### Predicting ICD Codes

```python
from src.predict_icd_base import predict_icd_codes

clinical_notes = [
    "Patient presents with fever and cough. History of diabetes.",
    "Patient has chest pain and shortness of breath. No prior cardiac history."
]

predictions = predict_icd_codes(
    clinical_notes=clinical_notes,
    model_folder_path="luqh/ClinicalT5-base"
)
```

### Training a Model

```python
from src.train_clinicalt5_moe import train_clinical_t5_moe
from src.ClinicalNoteDataset import ClinicalNoteDataset

# Create dataset
dataset = ClinicalNoteDataset(
    clinical_notes=notes,
    tokenizer=tokenizer,
    max_length=512
)

# Train model
model = train_clinical_t5_moe(
    model=model,
    train_dataloader=train_dataloader,
    valid_dataloader=valid_dataloader,
    num_epochs=3
)
```

## Model Architecture

The system uses T5 (Text-to-Text Transfer Transformer) as its base architecture, with three variants:

1. **Base Model**: Standard T5 encoder-decoder architecture
2. **MoE Model**: Replaces feed-forward layers with Mixture of Experts
3. **ST-MoE Model**: Implements sparse/switch transformer architecture for efficient computation

Input Format:
```
Clinical note: [CLINICAL TEXT] Predict ICD codes: <extra_id_0>
```

Output Format:
```
<extra_id_0> ICD-10:J18.9,E11.9,I10 <extra_id_1>
```

## Dependencies

Main dependencies include:
- Python 3.10+
- PyTorch
- Transformers
- pandas
- scikit-learn
- tqdm

See `environment.yml` for complete list.

## Notes

- Model weights and configuration files are automatically downloaded from Hugging Face
- The project uses Flax weights converted to PyTorch format
- Large model files are ignored by git (see .gitignore)

## License

[Add your license information here]
