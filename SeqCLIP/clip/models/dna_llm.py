import os
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
)

from typing import Optional, List, Dict, Any, Union, Tuple

from clip.utils.dna_utils import DNAInput
from clip.models.dl.processing_dl import DLProcessor
from clip.models.dl.chat_template_dl import CHAT_TEMPLATE

class DNALLMModel(nn.Module):
    """
    A combined model that processes both DNA sequences and text inputs.

    The model uses a DNA encoder (like NucleotideTransformer) to extract features from DNA sequences
    and a text model (LLM) to process text inputs and generate responses. The DNA features are
    projected to the text model's embedding space and prepended to the text embeddings.
    """

    def __init__(
        self,
        text_model_name: str,
        dna_model_name: str,
        cache_dir: Optional[str] = None,
        max_length_dna: int = 2048,
        max_length_text: int = 512,
        text_model_finetune: bool = True,
        dna_model_finetune: bool = True,
        offline: bool = False,
    ):
        """
        Initialize the DNALLMModel.

        Args:
            text_model_name: Name of the text model to be used.
            dna_model_name: Name of the DNA model to be used.
            cache_dir: Directory to cache the models.
            max_length_dna: Maximum length of DNA sequences. Defaults to 2048.
            max_length_text: Maximum length of text sequences. Defaults to 512.
            text_model_finetune: Whether to finetune the text model. Defaults to True.
            dna_model_finetune: Whether to finetune the DNA model. Defaults to True.
        """
        super().__init__()

        self.text_model_finetune = text_model_finetune
        self.dna_model_finetune = dna_model_finetune
        self.max_length_dna = max_length_dna
        self.max_length_text = max_length_text


        # Load the text model and tokenizer
        self.text_model = AutoModelForCausalLM.from_pretrained(
            text_model_name, cache_dir=cache_dir, trust_remote_code=True, local_files_only=offline,
        )
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name, cache_dir=cache_dir, trust_remote_code=True)
        self.text_config = self.text_model.config
        self.text_tokenizer.chat_template = CHAT_TEMPLATE
        self.text_tokenizer.pad_token = self.text_tokenizer.eos_token

        new_tokens = ["<|dna_start|>", "<|dna_pad|>", "<|dna_end|>"]
        self.text_tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
        self.dna_token_id = self.text_tokenizer.convert_tokens_to_ids("<|dna_pad|>")


        # Load the DNA model and tokenizer
        self.dna_model = AutoModelForMaskedLM.from_pretrained(
            dna_model_name, cache_dir=cache_dir, trust_remote_code=True, local_files_only=offline,
        )
        self.dna_tokenizer = AutoTokenizer.from_pretrained(dna_model_name, cache_dir=cache_dir, trust_remote_code=True)
        self.dna_config = self.dna_model.config


        # Get model dimensions
        self.text_hidden_size = self.text_config.hidden_size
        self.dna_hidden_size = self.dna_config.hidden_size

        # Create projection layer to map DNA embeddings to text model's embedding space
        self.dna_projection = nn.Linear(self.dna_hidden_size, self.text_hidden_size)

        # Create processor for handling inputs
        self.processor = DLProcessor(tokenizer=self.text_tokenizer, dna_tokenizer=self.dna_tokenizer)

    
    def load_pretrained_dna_encoder(self, clip_ckpt_path: str):
            print(f"-> Loading DNA Encoder weights from CLIP checkpoint: {clip_ckpt_path}")
            try:
                clip_ckpt = torch.load(clip_ckpt_path, map_location='cpu')
                
                state_dict = clip_ckpt.get('state_dict', clip_ckpt)
                
                dna_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith('model.dna_model.'):
                        new_k = k[len('model.'):] 
                    elif k.startswith('dna_model.'):
                        new_k = k
                    else:
                        continue
                    dna_state_dict[new_k] = v

                if not dna_state_dict:
                    raise ValueError("Could not find DNA Encoder keys in the CLIP checkpoint.")

                missing_keys, unexpected_keys = self.dna_model.load_state_dict(dna_state_dict, strict=False)
                
                for param in self.dna_model.parameters():
                    param.requires_grad = False
                print("Pretrained DNA encoder loaded and frozen.")

            except Exception as e:
                print(f"Warning: Failed to load DNA encoder weights. Error: {e}")
                print("The model will use randomly initialized or default pretrained DNA encoder weights.")


    def process_dna_embeddings(
        self,
        dna_tokenized: Dict[str, torch.Tensor],
        batch_idx_map: List[int],
        batch_size: int,
        actual_valid_lens: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        """
        Process DNA sequences to obtain embeddings.

        Args:
            dna_tokenized: Tokenized DNA sequences
            batch_idx_map: Mapping of each sequence to its batch item
            batch_size: Number of items in the batch

        Returns:
            List of tensor embeddings for each batch item
        """
        # Process all sequences to get DNA representations
        try:
            target_device = next(self.dna_model.parameters()).device
        except StopIteration:
            target_device = list(self.parameters())[0].device

        input_ids = dna_tokenized["input_ids"].to(target_device)
        attention_mask = dna_tokenized["attention_mask"].to(target_device)

        with torch.no_grad():
            # Standard HuggingFace model
            # Use existing code path for HF models
            outputs = self.dna_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            # Get the last hidden state
            hidden_states = outputs.hidden_states[-1]  # shape: [n_seqs, seq_len, hidden_dim]

        # Project all embeddings at once
        hidden_states = hidden_states.to(device=self.dna_projection.weight.device, dtype=self.dna_projection.weight.dtype)
        projected_states = self.dna_projection(hidden_states)

        # Group embeddings by batch item
        result = [[] for _ in range(batch_size)]

        # For each sequence, get its embeddings and add to appropriate batch result
        for seq_idx, batch_idx in enumerate(batch_idx_map):
            valid_length = attention_mask[seq_idx].sum().item()
            if actual_valid_lens is not None:
                valid_length = min(valid_length, actual_valid_lens[seq_idx])
            seq_embedding = projected_states[seq_idx, :valid_length]
            result[batch_idx].append(seq_embedding)

        # Concatenate embeddings for each batch item
        for i in range(batch_size):
            if result[i]:
                result[i] = torch.cat(result[i], dim=0)
            else:
                result[i] = torch.zeros((0, self.text_hidden_size), device=target_device, dtype=projected_states.dtype)

        return result

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        dna_tokenized: Optional[Dict[str, torch.Tensor]] = None,
        batch_idx_map: Optional[List[int]] = None,
        actual_valid_lens=None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate text based on DNA and text inputs.

        Args:
            input_ids: Input IDs (used if provided directly)
            attention_mask: Attention mask (used if provided directly)
            dna_tokenized: Tokenized DNA sequences (used if provided directly)
            batch_idx_map: Batch mapping for DNA sequences (used if provided directly)
            labels: Labels for supervised fine-tuning (used if provided directly)
            **kwargs: Additional arguments for generation

        Returns:
            Outputs from the text model
        """
        # Ensure required inputs are available
        if input_ids is None or attention_mask is None:
            raise ValueError("Either 'inputs' or 'input_ids'/'attention_mask' must be provided")

        batch_size = input_ids.shape[0]

        # Get text embeddings from the model's embedding layer
        text_inputs_embeds = self.text_model.get_input_embeddings()(input_ids)

        if dna_tokenized is not None and batch_idx_map is not None:
            batch_dna_embeds = self.process_dna_embeddings(dna_tokenized, batch_idx_map, batch_size, actual_valid_lens=actual_valid_lens,)

            mask = input_ids == self.dna_token_id

            n_dna_tokens = mask.sum().item()
            dna_embeds_flat = torch.cat(batch_dna_embeds, dim=0)
            n_dna_features = dna_embeds_flat.shape[0]

            if n_dna_features != n_dna_tokens:
                raise ValueError(
                    f"DNA features and DNA tokens do not match: features {n_dna_features}, tokens: {n_dna_tokens}"
                )

            # Ensure DNA embeddings have the same dtype as the text embeddings
            dna_embeds_flat = dna_embeds_flat.to(dtype=text_inputs_embeds.dtype)
            text_inputs_embeds[mask] = dna_embeds_flat

        # Forward pass through the text model (loss is computed if labels is provided)
        outputs = self.text_model(
            inputs_embeds=text_inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

        return outputs

    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        dna_tokenized: Optional[Dict[str, torch.Tensor]] = None,
        batch_idx_map: Optional[List[int]] = None,
        actual_valid_lens=None,
        **generation_kwargs,
    ) -> Union[torch.Tensor, List[str]]:
        """
        Generate text based on DNA and text inputs.

        Args:
            inputs: The preprocessed inputs from the processor (preferred method)
            batch_dna_sequences: List of lists of DNA sequences per batch item (legacy method)
            input_texts: List of input texts (legacy method)
            input_ids: Input IDs (used if provided directly)
            attention_mask: Attention mask (used if provided directly)
            dna_tokenized: Tokenized DNA sequences (used if provided directly)
            batch_idx_map: Batch mapping for DNA sequences (used if provided directly)
            **generation_kwargs: Additional arguments for generation

        Returns:
            Generated token IDs which can be decoded using the processor
        """
        # Ensure required inputs are available
        if input_ids is None or attention_mask is None:
            raise ValueError("Either 'inputs' or 'input_ids'/'attention_mask' must be provided")

        batch_size = input_ids.shape[0]

        # Get text embeddings from the model's embedding layer
        text_inputs_embeds = self.text_model.get_input_embeddings()(input_ids)

        if dna_tokenized is not None and batch_idx_map is not None:
            batch_dna_embeds = self.process_dna_embeddings(dna_tokenized, batch_idx_map, batch_size, actual_valid_lens=actual_valid_lens,)

            mask = input_ids == self.dna_token_id

            n_dna_tokens = mask.sum().item()
            dna_embeds_flat = torch.cat(batch_dna_embeds, dim=0)
            n_dna_features = dna_embeds_flat.shape[0]

            if n_dna_features != n_dna_tokens:
                raise ValueError(
                    f"DNA features and DNA tokens do not match: features {n_dna_features}, tokens: {n_dna_tokens}"
                )

            # Ensure DNA embeddings have the same dtype as the text embeddings
            dna_embeds_flat = dna_embeds_flat.to(dtype=text_inputs_embeds.dtype)
            text_inputs_embeds[mask] = dna_embeds_flat

        # Generation parameters may need adjustment based on model type
        with torch.no_grad():
            outputs = self.text_model.generate(
                inputs_embeds=text_inputs_embeds,
                attention_mask=attention_mask,
                use_cache=True,
                **generation_kwargs,
            )

        return outputs