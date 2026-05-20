import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM
import numpy as np


class BioCLIPModel(nn.Module):
    def __init__(self, dna_model_name: str, text_model_name: str, projection_dim: int, cache_dir: str = None, local_files_only=True):
        super().__init__()
        print(f"Loading DNA model: {dna_model_name}")
        self.dna_model = AutoModelForMaskedLM.from_pretrained(dna_model_name, cache_dir=cache_dir, trust_remote_code=True, local_files_only=local_files_only)

        print(f"Loading text model: {text_model_name}")
        self.text_model = AutoModelForCausalLM.from_pretrained(text_model_name, cache_dir=cache_dir, trust_remote_code=True, local_files_only=local_files_only)

        dna_hidden_size = self.dna_model.config.hidden_size
        text_hidden_size = self.text_model.config.hidden_size

        # Project DNA embeddings into the text model's embedding space
        target_projection_dim = text_hidden_size
        self.dna_projection = nn.Linear(dna_hidden_size, target_projection_dim)

        # Learnable logit scale (as in CLIP)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.fusion_mlp = None

    def encode_dna_pair(
        self,
        ref_ids: torch.Tensor,
        ref_mask: torch.Tensor,
        var_ids: torch.Tensor,
        var_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Encode a (reference, variant) DNA pair into a single embedding by averaging."""
        ref_embeds = self.encode_dna(ref_ids, ref_mask)
        var_embeds = self.encode_dna(var_ids, var_mask)
        return (ref_embeds + var_embeds) / 2

    def encode_dna(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode a DNA sequence into a fixed-size embedding."""
        outputs = self.dna_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            last_hidden_state = outputs.hidden_states[-1]
        elif hasattr(self.dna_model, "esm"):
            last_hidden_state = self.dna_model.esm(input_ids, attention_mask=attention_mask).last_hidden_state
        else:
            last_hidden_state = outputs.logits

        # Mean pooling over non-padding tokens
        pooled_output = ((last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) /
                         attention_mask.sum(-1).unsqueeze(-1))

        dna_embeds = self.dna_projection(pooled_output)
        return dna_embeds

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode a text sequence into a fixed-size embedding."""
        outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )

        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            last_hidden_state = outputs.hidden_states[-1]
        elif hasattr(self.text_model, "model"):
            last_hidden_state = self.text_model.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).last_hidden_state
        else:
            last_hidden_state = outputs.logits

        # Mean pooling over non-padding tokens
        text_embeds = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1, keepdim=True)

        if text_embeds.ndim == 1:
            text_embeds = text_embeds.unsqueeze(0)
        elif text_embeds.ndim == 2 and text_embeds.shape[1] == 1:
            text_embeds = text_embeds.T

        return text_embeds
