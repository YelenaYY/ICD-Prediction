import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from mixture_of_experts import Experts, MoE, HeirarchicalMoE


class ExpertAdapter(nn.Module):

    def __init__(self, embedding_file) -> None:
        
        # Expected file format:
        #   |----------------------------|-----------------------------------|----------|
        #   | x (clinical note)          | e (embedding)                     | y        |
        #   |----------------------------|-----------------------------------|----------|
        #   | Lorem ipsum dolor sit amet | 0.718, 0.281, 0.828, 0.459, 0.045 | ICD Code |
        #   |----------------------------|-----------------------------------|----------|

        self.df = pd.read_csv(embedding_file)

    def forward(self, x):

        df = self.df[self.df['x'] == x]
        embedding = df['e']
        return embedding


class MoeModel:

    def __init__(self, embedding_file:list[str]) -> None:

        self.experts = []
        for embedding_file in list:
            self.experts.append(ExpertAdapter(embedding_file))
        self.num_experts = len(self.experts)
        self.dim = len(self.df.columns)  # For now, assume number of columns corresponds to dime
        self.moe = MoE(
            dim = self.dim,
            num_experts = 16,
            hidden_dim = None,
            second_policy_train = 'random',
            second_policy_eval = 'random',
            second_threshold_train = 0.2,
            second_threshold_eval = 0.2,
            capacity_factor_train = 1.25,
            capacity_factor_eval = 2.,
            loss_coef = 1e-2,
            experts = self.experts)
    