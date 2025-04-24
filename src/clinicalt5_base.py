import re
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class BaseT5Model:

    def __init__(self, model_folder_path:str, embedding_file:list[str] = None, from_flax:bool = True) -> None:

        # T5
        tokenizer = AutoTokenizer.from_pretrained(model_folder_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_folder_path, from_flax=from_flax)

        model.save_pretrained("ClinicalT5-pytorch", safe_serialization=False)
        tokenizer.save_pretrained("ClinicalT5-pytorch")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("ClinicalT5-pytorch")
        self.model.to(self.device)


        self.device = torch.device("cpu")

    # Clinical-T5 structure:
    #    Input: Clinical note: [CLINICAL TEXT] Predict ICD codes: <extra_id_0>
    #    Label: <extra_id_0> ICD-10:J18.9,E11.9,I10 <extra_id_1>
    #   Prompt: Assign ICD codes to the following clinical note: [CLINICAL TEXT] Codes: <extra_id_0>
    def run_t5model(self, clinical_text:str):
        input_text = f"Clinical note: {clinical_text} Predict ICD codes: <extra_id_0>"
        
        # Tokenize input
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        
        # Generate output
        output = self.model.generate(
            input_ids=input_ids,
            max_length=50,  # Adjust based on expected output length
            num_beams=4,    # Beam search for better results
            early_stopping=True
        )
        
        # Decode output
        output_text = self.tokenizer.decode(output[0], skip_special_tokens=False)
        print(f'Model Output Text: {output_text}')
        
        # Extract ICD codes
        codes = re.search("<extra_id_0>(.*?)<extra_id_1>", output_text)
        if codes:
            parsed_codes = codes.group(1).strip()
            print(f"Predicted ICD Codes: {parsed_codes}")
            return parsed_codes
        return None

if __name__ == "__main__":
    model = BaseT5Model(model_folder_path="luqh/ClinicalT5-base", embedding_file=None, from_flax=True)
    model.run_t5model("Patient presents with fever and cough. History of diabetes.")

