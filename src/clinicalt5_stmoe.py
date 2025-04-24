import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
from transformers.models.t5.modeling_t5 import T5Block

from st_moe_pytorch import MoE, SparseMoEBlock, Expert


# Replaces standard T5 FF network with an MoE layer
class MoET5Block(nn.Module):
    def __init__(
        self,
        config,
        has_relative_attention_bias=False,
        num_experts=16,
        expert_hidden_mult=4,
        threshold_train=0.2,
        threshold_eval=0.2,
        gating_top_n=2,
        add_ff_before=False,
        add_ff_after=True
    ):
        super().__init__()        
        
        # Standard T5 block
        self.t5_block = T5Block(config, has_relative_attention_bias)
        
        # Dimensions
        d_model = config.d_model
        
        # MoE
        moe = MoE(
            dim=d_model,
            num_experts=num_experts,
            expert_hidden_mult=expert_hidden_mult,
            threshold_train=threshold_train,
            threshold_eval=threshold_eval,
            gating_top_n=gating_top_n,
            balance_loss_coef=1e-2,
            router_z_loss_coef=1e-3
        )
        self.moe_block = SparseMoEBlock(moe, add_ff_before=add_ff_before, add_ff_after=add_ff_after)
            
        # Delete original FF network (optional)
        # delattr(self.t5_block.layer[1], "DenseReluDense")
        
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # First part of T5Block (self-attention)
        layer_output = self.t5_block.layer[0](
            hidden_states,
            attention_mask=attention_mask
        )
        
        hidden_states = layer_output[0]
        
        # Run MoE instead of FF
        normalized_hidden_states = self.t5_block.layer[1].layer_norm(hidden_states)
        
        # Run MoE
        moe_output = self.moe_block(normalized_hidden_states)
        
        layer_output = self.t5_block.layer[1].dropout(moe_output.outputs)
        layer_output = hidden_states + layer_output
        
        outputs = (layer_output,) + layer_output[1:] + (moe_output.total_aux_loss,)
        return outputs
    

# Clinical-T5 with MoE
class ClinicalT5STMoE(nn.Module):
    def __init__(
        self, 
        model_name="luqh/ClinicalT5-base", 
        num_experts=16,
        expert_hidden_mult=4,
        threshold_train=0.2,
        threshold_eval=0.2,
        gating_top_n=2,
        add_ff_before=False,
        add_ff_after=True,
        moe_layer_indices=None,  # Which layers to replace with MoE
        from_flax=True,
        device=None
    ):
        super().__init__()
        
        self.config = T5Config.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, from_flax=from_flax)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        # Store device
        self.device = device
        
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
                expert_hidden_mult=expert_hidden_mult,
                threshold_train=threshold_train,
                threshold_eval=threshold_eval,
                gating_top_n=gating_top_n,
                add_ff_before=add_ff_before,
                add_ff_after=add_ff_after
            )
            
        # Replace decoder layers
        for i in self.decoder_moe_layers:
            has_relative_attention_bias = (i == 0)
            self.model.decoder.block[i] = MoET5Block(
                self.config,
                has_relative_attention_bias=has_relative_attention_bias,
                num_experts=num_experts,
                expert_hidden_mult=expert_hidden_mult,
                threshold_train=threshold_train,
                threshold_eval=threshold_eval,
                gating_top_n=gating_top_n,
                add_ff_before=add_ff_before,
                add_ff_after=add_ff_after
            )
        
        # Initialize model with dummy input if in meta state
        if device is not None:
            if hasattr(self.model, 'is_meta') and self.model.is_meta:
                # Create dummy input
                dummy_input = self.tokenizer("dummy input", return_tensors="pt")
                
                # Move model to device using to_empty
                self.model = self.model.to_empty(device=device)
                
                # Initialize with dummy input
                with torch.no_grad():
                    self.model(**dummy_input)
            else:
                # If not in meta state, just move to device
                self.model = self.model.to(device)
        
        # Add medical vocab
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
            
        #  Apply patches (use more elegant approach)
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


