import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional,List

class LoRALayer():
    def __init__(self, r:int,lora_alpha:int,lora_dropout_proba:float,merge_weights:bool):
        self.r = r
        self.lora_alpha = lora_alpha
        
        if lora_dropout_proba > 0. : self.lora_dropout = nn.Dropout(p=lora_dropout_proba)
        else: self.lora_dropout = lambda x: x

        self.merged = False
        self.merge_weights = merge_weights


class Embedding(nn.Embedding, LoRALayer):
    def __init__(
            self,
            num_embeddings: int,    # Number of unique tokens in the vocabulary
            embedding_dim: int,     # Size of each embedding vector
            r: int = 0,             # LoRA rank (low-rank decomposition size)
            lora_alpha: int = 1,    # Scaling factor for LoRA adaptation
            lora_dropout_prob:float=0.,
            merge_weights: bool = True, # Whether to merge LoRA weights into the base embedding
            **kwargs                # Additional arguments for nn.Embedding
        ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout_proba=lora_dropout_prob, merge_weights=merge_weights)

        # Trainable parametsr
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            self.scaling = self.lora_alpha / self.r

            # Freeze the pre-trained weight matrix
            self.weight.requires_grad = False

        self.reset_parameters()
    
    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.zeros_(self.lora_A) # Init A to zeros
            nn.init.normal_(self.lora_B) # Init B with normal distribution

    def train(self, mode:bool=True):
        """"
        In training (mode=True) :
            LoRA Updates are applied separately and not merged

        In Inference (mode=False):
            Merges LorA updates into the original embeddings
            Removes the need for separate lora_A and lora_B computations
        """
        nn.Embedding.train(self, mode)
        if mode: 
            """
            If LoRA weights were previously merged, undo the merge by
            subtracting the LoRA update.
            """
            if self.merge_weights and self.merged:
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A).transpose(0,1) * self.scaling
                self.merge = False
        
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A).transpose(0,1) * self.scaling
                self.merged = True
    
    def forward(self, x):
        