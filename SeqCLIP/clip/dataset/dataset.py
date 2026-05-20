"""
Dataset utilities for SeClip.

Provides loaders and collate functions for two datasets:
  - KEGG: pathway-level QA with reference/variant DNA sequence pairs
    (Stage 1 contrastive pre-training and Stage 2 LLM fine-tuning)
  - NT18: single-sequence DNA classification tasks
"""

import json
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from datasets import Dataset as HFDataset, load_from_disk
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizer

from clip.models.dl.processing_dl import DLProcessor


# ==============================================================================
# Utilities
# ==============================================================================

def truncate_dna(example: Dict[str, Any], truncate_dna_per_side: int = 1024) -> Dict[str, Any]:
    """Trim both ends of reference/variant sequences to at most truncate_dna_per_side bases."""
    for key in ["reference_sequence", "variant_sequence"]:
        sequence = example[key]
        if len(sequence) > 2 * truncate_dna_per_side + 8:
            example[key] = sequence[truncate_dna_per_side:-truncate_dna_per_side]
    return example


def torch_to_hf_dataset(torch_dataset: TorchDataset) -> HFDataset:
    """Convert a PyTorch Dataset to a Hugging Face Dataset."""
    if len(torch_dataset) == 0:
        return HFDataset.from_dict({})
    first_item = torch_dataset[0]
    data_dict = {k: [] for k in first_item.keys()} if isinstance(first_item, dict) else {"data": []}
    for i in range(len(torch_dataset)):
        item = torch_dataset[i]
        if isinstance(item, dict):
            for k in data_dict:
                data_dict[k].append(item[k])
        else:
            data_dict["data"].append(item)
    return HFDataset.from_dict(data_dict)


# ==============================================================================
# KEGG dataset
# ==============================================================================

class KEGGDataset(TorchDataset):
    """KEGG pathway QA dataset with reference/variant DNA sequence pairs."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.data = []

        print(f"Loading dataset from: {data_dir}")
        hf_dataset = load_from_disk(data_dir)
        if hasattr(hf_dataset, "keys"):
            split = "train" if "train" in hf_dataset else list(hf_dataset.keys())[0]
            hf_dataset = hf_dataset[split]

        for i, item in enumerate(hf_dataset):
            if isinstance(item, str):
                try:
                    item = json.loads(item)
                except Exception as e:
                    raise ValueError(f"Item {i} is not valid JSON: {item[:100]}") from e
            self.data.append(self._process_item(item))

    def _process_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        reasoning_data = item.get("reasoning", "")
        if isinstance(reasoning_data, dict):
            reasoning = "\n".join(reasoning_data.get("reasoning_steps", []))
        elif isinstance(reasoning_data, list):
            reasoning = "\n".join(reasoning_data)
        else:
            reasoning = str(reasoning_data)

        return {
            "question": item.get("question", ""),
            "answer": item.get("answer", "").lower().strip(),
            "reasoning": reasoning,
            "reference_sequence": item.get("reference_sequence", "").upper().strip(),
            "variant_sequence": item.get("variant_sequence", "").upper().strip(),
            "kegg_id": item.get("ID", ""),
        }

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]


def get_format_kegg_function(model_name: str) -> Any:
    """Return the KEGG format function for 'llm' or 'dna-llm' model type."""
    if model_name.lower() == "llm":
        return format_kegg_for_llm
    elif model_name.lower() == "dna-llm":
        return format_kegg_for_dna_llm
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


def format_kegg_for_dna_llm(example: Dict[str, Any]) -> Dict[str, Any]:
    """Format a KEGG example for DNA-LLM (reference + variant DNA as separate inputs)."""
    return {
        "prompt": [
            {
                "role": "user",
                "content": [
                    *({"type": "dna", "text": None} for _ in range(2)),
                    {"type": "text", "text": example["question"].strip()},
                ],
            },
            {
                "role": "assistant",
                "reasoning_content": example["reasoning"].strip(),
                "content": [{"type": "text", "text": f"Answer: {example['answer'].strip()}"}],
            },
        ],
        "dna_sequences": [example["reference_sequence"], example["variant_sequence"]],
        "answer": example["answer"],
    }


def format_kegg_for_llm(example: Dict[str, Any]) -> Dict[str, Any]:
    """Format a KEGG example for a text-only LLM (sequences embedded in the prompt)."""
    question = (
        f"Reference sequence: {example['reference_sequence']}\n"
        f"Variant sequence: {example['variant_sequence']}\n"
        f"Question: {example['question']}"
    )
    return {
        "prompt": [
            {
                "role": "user",
                "content": [
                    *({"type": "dna", "text": None} for _ in range(2)),
                    {"type": "text", "text": question.strip()},
                ],
            },
            {
                "role": "assistant",
                "reasoning_content": example["reasoning"].strip(),
                "content": [{"type": "text", "text": f"Answer: {example['answer'].strip()}"}],
            },
        ],
        "dna_sequences": ["", ""],
        "answer": example["answer"],
    }


def qwen_dna_collate_fn(
    examples: List[Dict],
    processor: DLProcessor,
    max_length_text: int,
    max_length_dna: int,
    return_answer_in_batch: bool = False,
) -> Dict:
    """
    Collate function for Stage 2 KEGG fine-tuning.

    Builds input_ids and labels where only assistant response tokens are unmasked.
    """
    prompts_text = [
        processor.tokenizer.apply_chat_template(
            ex["prompt"], add_generation_prompt=False, tokenize=False
        )
        for ex in examples
    ]
    batch_dna_sequences = [ex["dna_sequences"] for ex in examples]

    batch = processor(
        text=prompts_text,
        batch_dna_sequences=batch_dna_sequences,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False,
        max_length_text=max_length_text,
        max_length_dna=max_length_dna,
    )

    input_ids = batch["input_ids"]
    text_vocab_size = processor.tokenizer.vocab_size
    pad_token_id = processor.tokenizer.pad_token_id

    modal_type_ids = torch.zeros_like(input_ids, dtype=torch.long)
    modal_type_ids[input_ids >= text_vocab_size] = 2
    modal_type_ids[(input_ids < text_vocab_size) & (input_ids != pad_token_id) & (batch["attention_mask"] == 1)] = 1
    batch["modal_type_ids"] = modal_type_ids

    labels = torch.full_like(input_ids, -100)

    assistant_start_ids = processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    im_end_ids = processor.tokenizer.encode("<|im_end|>", add_special_tokens=False)
    assistant_tensor = torch.tensor(assistant_start_ids, device=input_ids.device)
    im_end_tensor = torch.tensor(im_end_ids, device=input_ids.device)
    a_len, e_len = len(assistant_start_ids), len(im_end_ids)

    for i in range(input_ids.shape[0]):
        ids = input_ids[i]
        seq_len = ids.size(0)

        starts = [
            pos + a_len
            for pos in range(seq_len - a_len + 1)
            if torch.all(ids[pos: pos + a_len] == assistant_tensor)
        ]
        ends = [
            pos
            for pos in range(seq_len - e_len + 1)
            if torch.all(ids[pos: pos + e_len] == im_end_tensor)
        ]

        for start in starts:
            valid_ends = [e for e in ends if e > start]
            end = min(valid_ends) if valid_ends else seq_len
            if start < end:
                labels[i, start:end] = ids[start:end]

    labels[input_ids == pad_token_id] = -100
    batch["labels"] = labels

    if return_answer_in_batch:
        batch["answer"] = [ex["answer"] for ex in examples]
        batch["question"] = [ex.get("question", "") for ex in examples]
        batch["reference_sequence"] = [ex.get("reference_sequence", "") for ex in examples]
        batch["reasoning"] = [ex.get("reasoning", "") for ex in examples]
        id_key = "kegg_id" if "kegg_id" in examples[0] else "sample_id" if "sample_id" in examples[0] else None
        if id_key:
            batch[id_key] = [ex.get(id_key, "") for ex in examples]

    return batch


# ==============================================================================
# NT18 dataset
# ==============================================================================

def load_dna_qa_tasks(data_dir: str, split: str = "train") -> HFDataset:
    """Load NT18 DNA classification tasks from a directory of JSON files."""
    data_path = Path(data_dir)
    all_samples = []

    for json_file in data_path.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("split") != split:
            continue

        task_name = data.get("task_name", json_file.stem)
        task_labels = data.get("labels", None)
        for sample in data.get("samples", []):
            sample["task_name"] = task_name
            sample["task_labels"] = task_labels
            all_samples.append(sample)

    return HFDataset.from_list(all_samples)


def format_nt18_example(example: Dict[str, Any]) -> Dict[str, Any]:
    """Format an NT18 example into the chat format expected by DNA-LLM."""
    dna_sequence = example["sequences"][0]
    task_labels = example.get("task_labels")
    user_text = example["exchanges"][0]["message"].replace("<DNA>", "<DNA_SEQ>")
    seq_label = example["seq_label"][0]

    prompt_content = []
    parts = user_text.split("<DNA_SEQ>")
    if len(parts) > 1:
        if parts[0].strip():
            prompt_content.append({"type": "text", "text": parts[0].strip()})
        prompt_content.append({"type": "dna", "text": None})
        if parts[1].strip():
            prompt_content.append({"type": "text", "text": parts[1].strip()})
    else:
        prompt_content.append({"type": "dna", "text": None})
        prompt_content.append({"type": "text", "text": user_text.strip()})

    label_instruction = (
        "\n\nYou must answer this task as a classification.\n"
        "Instructions:\n"
        "- Respond with exactly ONE label from the following options:\n"
    )
    for label in task_labels:
        label_instruction += f"- {label}\n"
    label_instruction += "- Do NOT provide explanations.\n- Do NOT add punctuation or extra text.\n\nAnswer:"
    prompt_content.append({"type": "text", "text": label_instruction})

    return {
        "prompt": [{"role": "user", "content": prompt_content}],
        "dna_sequences": [dna_sequence],
        "seq_label": seq_label,
        "task_labels": task_labels,
        "task_name": example.get("task_name", "unknown"),
    }


def nt18_collate_fn(
    examples: List[Dict],
    processor: DLProcessor,
    max_length_text: int,
    max_length_dna: int,
    return_answer_in_batch: bool = False,
) -> Dict:
    """Collate function for NT18 single-sequence DNA classification."""
    tok = processor.tokenizer
    prompts_text, batch_dna_sequences, seq_labels, task_names, task_labels_list = [], [], [], [], []

    for ex in examples:
        batch_dna_sequences.append([ex["dna_sequences"][0]])
        seq_labels.append(ex["seq_label"])
        task_names.append(ex["task_name"])
        task_labels_list.append(ex["task_labels"])
        prompts_text.append(
            tok.apply_chat_template(ex["prompt"], add_generation_prompt=True, tokenize=False)
        )

    batch = processor(
        text=prompts_text,
        batch_dna_sequences=batch_dna_sequences,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False,
        max_length_text=max_length_text,
        max_length_dna=max_length_dna,
    )

    input_ids = batch["input_ids"]
    labels = torch.full_like(input_ids, -100)

    assistant_id = tok.convert_tokens_to_ids("assistant")
    for i, ex in enumerate(examples):
        task_vocab = task_labels_list[i]
        y = seq_labels[i]
        if y < 0 or y >= len(task_vocab):
            continue
        target_ids = tok.encode(" " + task_vocab[y], add_special_tokens=False)
        pos = (input_ids[i] == assistant_id).nonzero(as_tuple=False)
        if pos.numel() == 0:
            continue
        start = pos[-1].item() + 1
        end = min(start + len(target_ids), input_ids.size(1))
        labels[i, start:end] = torch.tensor(target_ids[: end - start], device=input_ids.device)

    batch["labels"] = labels
    batch["seq_label"] = torch.tensor(seq_labels, device=input_ids.device)
    batch["task_names"] = task_names
    batch["task_labels"] = task_labels_list

    return batch


# ==============================================================================
# Stage 1: contrastive collate
# ==============================================================================

def contrastive_collate_fn(
    batch: List[Dict],
    dna_tokenizer: PreTrainedTokenizer,
    text_tokenizer: PreTrainedTokenizer,
    max_length_dna: int,
    max_length_text: int,
) -> Dict[str, torch.Tensor]:
    """
    Collate function for Stage 1 CLIP contrastive pre-training.

    Tokenizes the structured text description and both the reference and
    variant DNA sequences from each sample.

    Each sample must contain 'structured_text', 'reference_sequence',
    and 'variant_sequence'.
    """
    if not batch or "structured_text" not in batch[0]:
        raise ValueError("Each sample must contain 'structured_text', 'reference_sequence', and 'variant_sequence'.")

    text_inputs = text_tokenizer(
        [item["structured_text"] for item in batch],
        padding="max_length", truncation=True, max_length=max_length_text, return_tensors="pt",
    )
    ref_inputs = dna_tokenizer(
        [item["reference_sequence"] for item in batch],
        padding="max_length", truncation=True, max_length=max_length_dna, return_tensors="pt",
    )
    var_inputs = dna_tokenizer(
        [item["variant_sequence"] for item in batch],
        padding="max_length", truncation=True, max_length=max_length_dna, return_tensors="pt",
    )

    return {
        "text_input_ids": text_inputs.input_ids,
        "text_attention_mask": text_inputs.attention_mask,
        "ref_input_ids": ref_inputs.input_ids,
        "ref_attention_mask": ref_inputs.attention_mask,
        "var_input_ids": var_inputs.input_ids,
        "var_attention_mask": var_inputs.attention_mask,
    }
