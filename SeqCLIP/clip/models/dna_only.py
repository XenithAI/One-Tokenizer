import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from transformers import AutoModelForMaskedLM, AutoTokenizer


class SelfAttentionPooling(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
        )
        self.query = nn.Parameter(torch.randn(1, 1, hidden_size))

    def forward(self, embeddings, attention_mask=None):
        batch_size = embeddings.size(0)
        query = self.query.expand(batch_size, -1, -1)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        context, _ = self.attention(
            query=query,
            key=embeddings,
            value=embeddings,
            key_padding_mask=key_padding_mask,
        )
        return context.squeeze(1)


class DNAClassifierModel(nn.Module):
    """
    DNA sequence pair classifier.

    Encodes reference and alternate sequences with a pretrained DNA model,
    pools them with learned self-attention, concatenates the pair, and
    classifies via a two-layer MLP head.
    """

    def __init__(
        self,
        dna_model_name: str,
        cache_dir: str = None,
        max_length_dna: int = 4096,
        num_classes: int = 2,
        dna_is_evo2: bool = False,
        dna_embedding_layer: str = None,
        train_just_classifier: bool = True,
    ):
        super().__init__()

        self.dna_model_name = dna_model_name
        self.cache_dir = cache_dir
        self.max_length_dna = max_length_dna
        self.num_classes = num_classes
        self.dna_is_evo2 = dna_is_evo2
        self.dna_embedding_layer = dna_embedding_layer
        self.train_just_classifier = train_just_classifier

        if not self.dna_is_evo2:
            self.dna_model = AutoModelForMaskedLM.from_pretrained(
                dna_model_name, cache_dir=cache_dir, trust_remote_code=True
            )
            self.dna_tokenizer = AutoTokenizer.from_pretrained(dna_model_name, trust_remote_code=False)
            self.dna_config = self.dna_model.config
        else:
            from evo2 import Evo2
            from clip.models.evo2_tokenizer import Evo2Tokenizer
            self.dna_model = Evo2(dna_model_name)
            self.dna_tokenizer = Evo2Tokenizer(self.dna_model.tokenizer)
            self.dna_config = self.dna_model.model.config

        self.hidden_size = self.dna_config.hidden_size
        self.pooler = SelfAttentionPooling(self.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, num_classes),
        )

    def get_dna_embedding(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Pool a single DNA sequence into a fixed-size embedding."""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        elif attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)

        with torch.set_grad_enabled(not self.train_just_classifier):
            if self.dna_is_evo2 and self.dna_embedding_layer is not None:
                _, embeddings = self.dna_model(
                    input_ids,
                    return_embeddings=True,
                    layer_names=[self.dna_embedding_layer],
                )
                hidden_states = embeddings[self.dna_embedding_layer]
            else:
                outputs = self.dna_model(
                    input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_states = outputs.hidden_states[-1]

        return self.pooler(hidden_states, attention_mask).squeeze(0)

    def forward(self, ref_ids=None, alt_ids=None, ref_attention_mask=None, alt_attention_mask=None):
        batch_size = ref_ids.shape[0] if ref_ids is not None else alt_ids.shape[0]

        ref_embeddings = [self.get_dna_embedding(ref_ids[i], ref_attention_mask[i]) for i in range(batch_size)]
        alt_embeddings = [self.get_dna_embedding(alt_ids[i], alt_attention_mask[i]) for i in range(batch_size)]

        ref_embeddings = torch.stack(ref_embeddings)
        alt_embeddings = torch.stack(alt_embeddings)

        combined = torch.cat([ref_embeddings, alt_embeddings], dim=1)
        return self.classifier(combined)
