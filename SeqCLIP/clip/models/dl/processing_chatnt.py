# processing_chatnt.py
from typing import List, Optional, Union, Dict, Any
import torch
from transformers.processing_utils import ProcessorMixin

class ChatNTProcessor(ProcessorMixin):
    attributes = ["tokenizer", "dna_tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(self, tokenizer, dna_tokenizer, **kwargs):
        super().__init__(tokenizer, dna_tokenizer, **kwargs)
        self.tokenizer = tokenizer
        self.dna_tokenizer = dna_tokenizer

    def __call__(
        self,
        text: Union[str, List[str]] = None,
        batch_dna_sequences: List[List[str]] = None,
        max_length_text: int = 512,
        max_length_dna: int = 512,
        padding: bool = True,
        return_tensors: str = "pt",
        **kwargs
    ):
        encoding = self.tokenizer(
            text, 
            max_length=max_length_text, 
            padding=padding, 
            truncation=True, 
            return_tensors=return_tensors
        )

        if batch_dna_sequences is not None:
            dna_list, batch_idx_map = [], []
            for i, seqs in enumerate(batch_dna_sequences):
                for s in seqs:
                    dna_list.append(s)
                    batch_idx_map.append(i)

            dna_encoding = self.dna_tokenizer(
                dna_list, 
                max_length=max_length_dna, 
                padding=True, 
                truncation=True, 
                return_tensors=return_tensors
            )
            encoding["dna_tokenized"] = dna_encoding
            encoding["batch_idx_map"] = torch.tensor(batch_idx_map)

        return encoding

    def batch_decode(self, *args, **kwargs): return self.tokenizer.batch_decode(*args, **kwargs)
    def decode(self, *args, **kwargs): return self.tokenizer.decode(*args, **kwargs)