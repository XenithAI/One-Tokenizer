from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding
from transformers.utils import logging

logger = logging.get_logger(__name__)


class Evo2Tokenizer(PreTrainedTokenizer):
    """HuggingFace-compatible wrapper around the Evo2 CharLevelTokenizer."""

    vocab_files_names = {}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        evo2_tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
        **kwargs,
    ):
        self.evo2_tokenizer = evo2_tokenizer
        self._pad_token = pad_token
        self._eos_token = eos_token
        self._bos_token = bos_token
        self._unk_token = unk_token
        super().__init__(bos_token=bos_token, eos_token=eos_token, pad_token=pad_token, unk_token=unk_token, **kwargs)
        self.pad_token_id = self.evo2_tokenizer.pad_id
        self.eos_token_id = self.evo2_tokenizer.eos_id

    @property
    def vocab_size(self) -> int:
        return self.evo2_tokenizer.vocab_size

    def get_vocab(self) -> Dict:
        return {chr(i): i for i in range(self.vocab_size)}

    def _tokenize(self, text: str) -> List[int]:
        return [chr(int(token)) for token in self.evo2_tokenizer.tokenize(text)]

    def _convert_token_to_id(self, token: str) -> int:
        return ord(token)

    def _convert_id_to_token(self, index: int) -> str:
        return chr(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return "".join(tokens)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        return ()

    def __call__(
        self,
        text: Union[str, List[str]],
        text_pair: Optional[Union[str, List[str]]] = None,
        padding: Union[bool, str] = False,
        truncation: Union[bool, str] = False,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(text, str):
            text = [text]

        input_ids_list = []
        for seq in text:
            tokens = [int(t) for t in self.evo2_tokenizer.tokenize(seq)]
            if truncation and max_length and len(tokens) > max_length:
                tokens = tokens[:max_length]
            input_ids_list.append(tokens)

        if padding:
            max_len = max(len(ids) for ids in input_ids_list)
            padded_input_ids, attention_mask = [], []
            for ids in input_ids_list:
                pad_len = max_len - len(ids)
                padded_input_ids.append([self.pad_token_id] * pad_len + ids)
                attention_mask.append([0] * pad_len + [1] * len(ids))
            input_ids_list = padded_input_ids
        else:
            attention_mask = [[1] * len(ids) for ids in input_ids_list]

        result = {"input_ids": input_ids_list}
        if return_attention_mask:
            result["attention_mask"] = attention_mask

        if return_tensors == "pt":
            result = {k: torch.tensor(v) for k, v in result.items()}

        return BatchEncoding(data=result, tensor_type=return_tensors, prepend_batch_axis=False, encoding=None)

    def batch_decode(
        self,
        sequences: Union[List[int], List[List[int]], torch.Tensor],
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> List[str]:
        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()
        return self.evo2_tokenizer.detokenize_batch(sequences)

    def decode(
        self,
        token_ids: Union[int, List[int], torch.Tensor],
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if not isinstance(token_ids, list) or not token_ids or not isinstance(token_ids[0], (list, torch.Tensor)):
            return self.evo2_tokenizer.detokenize(token_ids)
        return self.batch_decode(token_ids, skip_special_tokens, **kwargs)[0]


def register_evo2_tokenizer():
    """Register Evo2Tokenizer with HuggingFace's AutoTokenizer."""
    AutoTokenizer.register("evo2", Evo2Tokenizer)
