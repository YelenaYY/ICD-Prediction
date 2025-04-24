import re
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer
from transformers.models.t5.modeling_t5 import T5Block
from mixture_of_experts import Experts, MoE, HeirarchicalMoE


# Replaces standard T5 FF network with an MoE layer
class MoET5Block(nn.Module):

    def __init__(
        self,
        config,
        has_relative_attention_bias=False,
        num_experts=16,
        use_hierarchical=False
    ):
        super().__init__()
        
        # Standard T5 block
        self.t5_block = T5Block(config, has_relative_attention_bias)
        
        # Dimensions
        d_model = config.d_model
        d_ff = config.d_ff
        
        # Replace FF layer with MoE
        if use_hierarchical:
            self.moe = HeirarchicalMoE(
                dim=d_model,
                num_experts=(4, 4) if num_experts == 16 else num_experts,
                hidden_dim=d_ff,
                activation=nn.ReLU(),  # T5 typically uses ReLU
                loss_coef=0.01
            )
        else:
            self.moe = MoE(
                dim=d_model,
                num_experts=num_experts,
                hidden_dim=d_ff,
                activation=nn.ReLU,
                loss_coef=0.01
            )
            
        # Delete original FF network (optional)
        # delattr(self.t5_block.layer[1], "DenseReluDense")
        
    def forward(self, *args, **kwargs):
        # Rirst part of T5Block (self-attention)
        hidden_states = args[0]
        attention_mask = kwargs.get("attention_mask", None)
        
        # First layer (self-attention)
        layer_output = self.t5_block.layer[0](
            hidden_states,
            attention_mask=attention_mask
        )
        
        hidden_states = layer_output[0]
        
        # Run MoE
        normalized_hidden_states = self.t5_block.layer[1].layer_norm(hidden_states)
        moe_output, moe_loss = self.moe(normalized_hidden_states)
        
        layer_output = self.t5_block.layer[1].dropout(moe_output)
        layer_output = hidden_states + layer_output
        
        outputs = (layer_output,) + layer_output[1:] + (moe_loss,)
        return outputs


# Clinical-T5 with MoE
class ClinicalT5MoE(nn.Module):

    def __init__(
        self, 
        model_name="t5-base", 
        num_experts=16,
        use_hierarchical=False,
        moe_layer_indices=None,  # Layers to replace with MoE
        from_flax=True
    ):
        super().__init__()
        
        self.config = T5Config.from_pretrained(model_name)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, from_flax=from_flax)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        # Replace every other layer with MoE by default
        if moe_layer_indices is None:
            encoder_layers = self.config.num_layers
            decoder_layers = self.config.num_decoder_layers
            self.encoder_moe_layers = list(range(0, encoder_layers, 2))
            self.decoder_moe_layers = list(range(0, decoder_layers, 2))
        else:
            self.encoder_moe_layers, self.decoder_moe_layers = moe_layer_indices
        
        # Replace encoder layers
        for i in self.encoder_moe_layers:
            has_relative_attention_bias = (i == 0)
            self.model.encoder.block[i] = MoET5Block(
                self.config,
                has_relative_attention_bias=has_relative_attention_bias,
                num_experts=num_experts,
                use_hierarchical=use_hierarchical
            )
            
        # Replace decoder layers
        for i in self.decoder_moe_layers:
            has_relative_attention_bias = (i == 0)
            self.model.decoder.block[i] = MoET5Block(
                self.config,
                has_relative_attention_bias=has_relative_attention_bias,
                num_experts=num_experts,
                use_hierarchical=use_hierarchical
            )
            
        # Add medical vocab (optional, recommended)
        # self.add_medical_tokens()
        

    def add_medical_tokens(self, medical_tokens=None):
        tokenizer = T5Tokenizer.from_pretrained(self.config._name_or_path)
        
        # Example
        if medical_tokens is None:
            medical_tokens = [
                "ICD-10:", "CPT:", "SNOMED:", "LOINC:", "RxNorm:",
                "<diagnosis>", "</diagnosis>", "<procedure>", "</procedure>",
                "<medication>", "</medication>", "<lab>", "</lab>"
            ]
        
        num_added = tokenizer.add_tokens(medical_tokens)
        if num_added > 0:
            self.model.resize_token_embeddings(len(tokenizer))
            
        self.tokenizer = tokenizer

        
    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, 
                labels=None, **kwargs):
        
        moe_losses = []
        
        original_encoder_block_forward = self.model.encoder.block[0].forward
        original_decoder_block_forward = self.model.decoder.block[0].forward
        
        def encoder_block_forward_with_loss_collection(self, *args, **kwargs):
            outputs = original_encoder_block_forward(*args, **kwargs)
            if len(outputs) > 2 and isinstance(outputs[-1], torch.Tensor):
                moe_losses.append(outputs[-1])
            return outputs
            
        def decoder_block_forward_with_loss_collection(self, *args, **kwargs):
            outputs = original_decoder_block_forward(*args, **kwargs)
            if len(outputs) > 2 and isinstance(outputs[-1], torch.Tensor):
                moe_losses.append(outputs[-1])
            return outputs
            
        # Apply patches (use more elegant approach)
        for i in self.encoder_moe_layers:
            self.model.encoder.block[i].forward = encoder_block_forward_with_loss_collection.__get__(
                self.model.encoder.block[i], type(self.model.encoder.block[i])
            )
            
        for i in self.decoder_moe_layers:
            self.model.decoder.block[i].forward = decoder_block_forward_with_loss_collection.__get__(
                self.model.decoder.block[i], type(self.model.decoder.block[i])
            )
            
        # Forward
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            **kwargs
        )
        
        # Sum MoE loss to output
        if moe_losses:
            total_moe_loss = torch.stack(moe_losses).sum()
            if hasattr(outputs, "loss") and outputs.loss is not None:
                outputs.loss = outputs.loss + total_moe_loss
                
        return outputs

