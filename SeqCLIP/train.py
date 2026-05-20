"""
SeClip two-stage training pipeline.

Stage 1 (--stage clip):
    Contrastive pre-training of BioCLIPModel — aligns DNA sequence embeddings
    with text embeddings using InfoNCE loss. The trained DNA encoder is saved
    for use in Stage 2.

Stage 2 (--stage llm):
    Supervised fine-tuning of DNALLMModel on a generation task (e.g. KEGG QA),
    initializing the DNA encoder from the Stage 1 checkpoint.

Usage examples:
    # Stage 1: CLIP pre-training on KEGG
    python train.py --stage clip \\
        --text_model_name Qwen/Qwen3-1.7B \\
        --dna_model_name InstaDeepAI/nucleotide-transformer-v2-500m-multi-species \\
        --dataset_type kegg --kegg_data_dir /path/to/kegg

    # Stage 2: LLM fine-tuning with pre-trained DNA encoder
    python train.py --stage llm \\
        --text_model_name /path/to/qwen3_1.7b \\
        --dna_model_name /path/to/nt_model \\
        --clip_dna_model_path /path/to/clip_checkpoint.ckpt \\
        --task_data_dir /path/to/kegg/train \\
        --val_data_dir /path/to/kegg/val
"""

import os
import time
from argparse import ArgumentParser
from functools import partial
from typing import Dict, List, Optional

import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from datasets import concatenate_datasets, load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import precision_recall_fscore_support
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from transformers.tokenization_utils_base import BatchEncoding

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DeepSpeedStrategy

from clip.dataset import KEGGDataset, format_kegg_for_dna_llm, qwen_dna_collate_fn, get_format_kegg_function, contrastive_collate_fn
from clip.models.dna_clip import BioCLIPModel
from clip.models.dna_llm import DNALLMModel
from clip.models.dl.processing_dl import DLProcessor


torch.multiprocessing.set_sharing_strategy("file_system")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ==============================================================================
# Stage 1: CLIP contrastive pre-training
# ==============================================================================

class DNACLIPFineTuner(pl.LightningModule):
    """
    PyTorch Lightning module for contrastive pre-training of BioCLIPModel.

    Aligns DNA sequence embeddings with text embeddings using InfoNCE (CLIP) loss.
    Both a reference and a variant DNA sequence are encoded per sample; their
    embeddings are averaged to produce a single DNA representation.
    """

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.model = BioCLIPModel(
            dna_model_name=hparams.dna_model_name,
            text_model_name=hparams.text_model_name,
            projection_dim=hparams.projection_dim,
            cache_dir=hparams.cache_dir,
        )

        self.dna_tokenizer = AutoTokenizer.from_pretrained(
            hparams.dna_model_name, cache_dir=hparams.cache_dir, trust_remote_code=True
        )
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            hparams.text_model_name, cache_dir=hparams.cache_dir, trust_remote_code=True
        )
        if self.text_tokenizer.pad_token is None:
            self.text_tokenizer.pad_token = self.text_tokenizer.eos_token

        self._prep_for_training()

    def _get_target_modules(self) -> List[str]:
        """Collect linear layer names in the text model for LoRA targeting."""
        seen = set()
        target_modules = []
        for name, module in self.model.text_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                last = name.split(".")[-1]
                if last != "lm_head" and last not in seen:
                    target_modules.append(last)
                    seen.add(last)
        for pattern in ["q_proj", "k_proj", "v_proj", "out_proj", "query", "key", "value"]:
            if pattern not in seen:
                target_modules.append(pattern)
        return target_modules

    def _prep_for_training(self):
        """Configure which parameters are trainable."""
        if not self.hparams.dna_model_finetune:
            for param in self.model.dna_model.parameters():
                param.requires_grad = False

        if self.hparams.text_model_finetune:
            lora_config = LoraConfig(
                r=self.hparams.lora_rank,
                lora_alpha=self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout,
                target_modules=self._get_target_modules(),
                init_lora_weights="gaussian",
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model.text_model = prepare_model_for_kbit_training(self.model.text_model)
            self.model.text_model = get_peft_model(self.model.text_model, lora_config)
        else:
            for param in self.model.text_model.parameters():
                param.requires_grad = False

        for param in self.model.dna_projection.parameters():
            param.requires_grad = True

    def _step(self, batch: Dict, prefix: str) -> torch.Tensor:
        dna_embeds = self.model.encode_dna_pair(
            ref_ids=batch["ref_input_ids"],
            ref_mask=batch["ref_attention_mask"],
            var_ids=batch["var_input_ids"],
            var_mask=batch["var_attention_mask"],
        )
        text_embeds = self.model.encode_text(
            input_ids=batch["text_input_ids"],
            attention_mask=batch["text_attention_mask"],
        )

        dna_embeds = F.normalize(dna_embeds, p=2, dim=1)
        text_embeds = F.normalize(text_embeds, p=2, dim=1)

        logit_scale = torch.clamp(self.model.logit_scale, 0, 4.6).exp()
        logits_per_dna = torch.matmul(dna_embeds, text_embeds.t()) * logit_scale
        logits_per_text = logits_per_dna.t()

        labels = torch.arange(dna_embeds.shape[0], device=self.device)
        loss = (F.cross_entropy(logits_per_dna, labels) + F.cross_entropy(logits_per_text, labels)) / 2

        self.log(f"{prefix}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_after_backward(self):
        self.model.logit_scale.data.clamp_(0, 4.6)

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _format_for_contrastive(self, example: Dict) -> Dict:
        """Strip the question suffix to isolate the structured description text."""
        text = example.get("question", "")
        for delim in [
            "\n\nGiven this context,", "\n\nWhat is the KEGG function?",
            "\nGiven this context,", "\nWhat is the KEGG function?",
        ]:
            if delim in text:
                text = text.split(delim)[0]
                break
        if "reference_sequence" not in example or "variant_sequence" not in example:
            raise ValueError(f"Missing DNA sequences. Keys: {list(example.keys())}")
        return {
            "structured_text": text.strip(),
            "reference_sequence": example["reference_sequence"],
            "variant_sequence": example["variant_sequence"],
        }

    def _load_dataset(self, split: str):
        num_proc = self.hparams.num_workers or None
        dtype = self.hparams.dataset_type

        def safe_load(path):
            try:
                return load_from_disk(path)
            except Exception as e:
                raise RuntimeError(f"Failed to load dataset from: {path}") from e

        if dtype == "kegg":
            ds = safe_load(self.hparams.kegg_data_dir)
            ds = ds[split] if hasattr(ds, "keys") else ds
            ds = ds.map(get_format_kegg_function(self.hparams.model_type), num_proc=num_proc)

            if split == "val" and self.hparams.merge_val_test_set:
                all_ds = load_dataset(self.hparams.kegg_data_dir)
                fmt = get_format_kegg_function(self.hparams.model_type)
                ds = concatenate_datasets([
                    all_ds["test"].map(fmt, num_proc=num_proc),
                    all_ds["val"].map(fmt, num_proc=num_proc),
                ])

        else:
            raise ValueError(f"Unknown dataset type: {dtype}")

        return ds.map(self._format_for_contrastive, remove_columns=ds.column_names, num_proc=num_proc)

    def _make_dataloader(self, split: str, shuffle: bool) -> DataLoader:
        dataset = self._load_dataset(split)
        collate_fn = partial(
            contrastive_collate_fn,
            dna_tokenizer=self.dna_tokenizer,
            text_tokenizer=self.text_tokenizer,
            max_length_dna=self.hparams.max_length_dna,
            max_length_text=self.hparams.max_length_text,
        )
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=self.hparams.num_workers,
            persistent_workers=self.hparams.num_workers > 0,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self._make_dataloader("train", shuffle=True)

    def val_dataloader(self):
        return self._make_dataloader("val", shuffle=False)

    def test_dataloader(self):
        return self._make_dataloader("val", shuffle=False)


# ==============================================================================
# Stage 2: LLM fine-tuning with CLIP-pretrained DNA encoder
# ==============================================================================

class DNALLMFineTuner(pl.LightningModule):
    """
    PyTorch Lightning module for fine-tuning DNALLMModel on a generation task.

    The DNA encoder is initialized from a CLIP pre-training checkpoint
    (--clip_dna_model_path) and optionally frozen while the LLM is fine-tuned
    with LoRA. Evaluation runs full generation and reports accuracy/F1.
    """

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.model = DNALLMModel(
            text_model_name=hparams.text_model_name,
            dna_model_name=hparams.dna_model_name,
            cache_dir=hparams.cache_dir,
            max_length_dna=hparams.max_length_dna,
            max_length_text=hparams.max_length_text,
            text_model_finetune=hparams.text_model_finetune,
            dna_model_finetune=hparams.dna_model_finetune,
            offline=hparams.offline,
        )

        if hparams.clip_dna_model_path:
            self.model.load_pretrained_dna_encoder(hparams.clip_dna_model_path)

        self._prep_for_training()

        for arg in ["temperature", "top_p", "top_k", "do_sample"]:
            if hasattr(self.model.text_model.generation_config, arg):
                setattr(self.model.text_model.generation_config, arg, None)

    def _get_target_modules(self) -> List[str]:
        """Collect linear layer names in the text model for LoRA targeting."""
        seen = set()
        target_modules = []
        for name, module in self.model.text_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                last = name.split(".")[-1]
                if last != "lm_head" and last not in seen:
                    target_modules.append(last)
                    seen.add(last)
        for pattern in ["q_proj", "k_proj", "v_proj", "out_proj",
                        "query_key_value", "dense", "gate_proj", "up_proj", "down_proj"]:
            if pattern not in seen:
                target_modules.append(pattern)
        return list(set(target_modules))

    def _prep_for_training(self):
        """Configure which parameters are trainable."""
        if self.hparams.text_model_finetune:
            lora_config = LoraConfig(
                r=self.hparams.lora_rank,
                lora_alpha=self.hparams.lora_alpha,
                lora_dropout=self.hparams.lora_dropout,
                target_modules=self._get_target_modules(),
                init_lora_weights="gaussian",
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model.text_model = prepare_model_for_kbit_training(self.model.text_model)
            self.model.text_model = get_peft_model(self.model.text_model, lora_config)
            print("LLM LoRA applied.")
            self.model.text_model.print_trainable_parameters()
        else:
            for param in self.model.text_model.parameters():
                param.requires_grad = False
            print("LLM frozen.")

        if self.hparams.dna_model_finetune:
            for param in self.model.dna_model.parameters():
                param.requires_grad = True
            print("DNA encoder set to trainable.")
        else:
            for param in self.model.dna_model.parameters():
                param.requires_grad = False
            print("DNA encoder frozen.")

        for param in self.model.dna_projection.parameters():
            param.requires_grad = True
        print("DNA projection layer set to trainable.")

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Move all tensors (including nested BatchEncoding) to the target device."""
        def move(obj):
            if isinstance(obj, torch.Tensor):
                return obj.to(device)
            if isinstance(obj, BatchEncoding):
                return {k: move(v) for k, v in obj.items()}
            if isinstance(obj, dict):
                return {k: move(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [move(v) for v in obj]
            return obj
        return move(batch)

    def _step(self, batch: Dict, prefix: str) -> torch.Tensor:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device)
        if "dna_tokenized" in batch:
            for k, v in batch["dna_tokenized"].items():
                if isinstance(v, torch.Tensor):
                    batch["dna_tokenized"][k] = v.to(self.device)

        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            dna_tokenized=batch.get("dna_tokenized"),
            batch_idx_map=batch.get("batch_idx_map"),
            labels=batch.get("labels"),
        )
        loss = outputs.loss
        self.log(f"{prefix}_loss", loss, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        if prefix != "test":
            try:
                self.log("lr", self.lr_schedulers().get_last_lr()[0], on_step=True, prog_bar=True, logger=True)
            except Exception:
                pass
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        pass  # evaluation runs in on_test_epoch_end

    def on_test_epoch_end(self):
        if self.global_rank != 0:
            return

        wandb_logger = self.logger.experiment if isinstance(self.logger, WandbLogger) else None
        print("\nRunning generation evaluation on test set...")
        self.model.eval()

        dataloader = self.trainer.test_dataloaders
        if isinstance(dataloader, list):
            dataloader = dataloader[0]

        total, correct = 0, 0
        detailed_results = []
        y_true, y_pred = [], []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                batch_labels = batch["labels"].to(self.device)

                questions = batch.get("question", [""] * len(input_ids))
                answers = batch.get("answer", [""] * len(input_ids))
                ref_seqs = batch.get("reference_sequence", [""] * len(input_ids))
                reasonings = batch.get("reasoning", [""] * len(input_ids))

                dna_tokenized = batch.get("dna_tokenized")
                if dna_tokenized:
                    dna_tokenized = {k: v.to(self.device) for k, v in dna_tokenized.items()}
                batch_idx_map = batch.get("batch_idx_map")

                for i in range(len(input_ids)):
                    answer_indices = (batch_labels[i] != -100).nonzero()
                    prompt_end = (
                        answer_indices[0].item() if len(answer_indices) > 0
                        else attention_mask[i].sum().item()
                    )
                    prompt_ids = input_ids[i, :prompt_end]
                    prompt_mask = attention_mask[i, :prompt_end]

                    curr_dna, curr_map = None, None
                    if dna_tokenized is not None and batch_idx_map is not None:
                        idxs = [j for j, v in enumerate(batch_idx_map) if v == i]
                        if idxs:
                            curr_dna = {k: dna_tokenized[k][idxs] for k in dna_tokenized}
                            curr_map = [0] * len(idxs)

                    gen_output = self.model.generate(
                        input_ids=prompt_ids.unsqueeze(0),
                        attention_mask=prompt_mask.unsqueeze(0),
                        dna_tokenized=curr_dna,
                        batch_idx_map=curr_map,
                        max_new_tokens=512,
                        do_sample=True,
                        repetition_penalty=1.1,
                        temperature=0.7,
                        top_p=0.9,
                    )

                    gen_text = self.model.text_tokenizer.decode(gen_output[0], skip_special_tokens=True)
                    gen_text = gen_text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()

                    true_answer = str(answers[i]).strip()
                    is_correct = true_answer.lower() in gen_text.lower()
                    y_true.append(true_answer.lower())
                    y_pred.append(true_answer.lower() if is_correct else gen_text.lower())
                    total += 1
                    correct += int(is_correct)

                    detailed_results.append({
                        "question": questions[i][:200],
                        "reference_sequence": ref_seqs[i][:50],
                        "generated_response": gen_text,
                        "true_answer": true_answer,
                        "reasoning": reasonings[i] if i < len(reasonings) else "",
                        "is_correct": is_correct,
                    })
                    if total % 20 == 0:
                        print(f"[{total}] GT: {true_answer} | Match: {is_correct}")

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        acc = correct / total if total > 0 else 0.0
        self.log_dict(
            {"test_acc": acc, "test_precision": precision, "test_recall": recall, "test_f1": f1},
            rank_zero_only=True,
        )

        os.makedirs(self.hparams.checkpoint_dir, exist_ok=True)
        csv_path = os.path.join(self.hparams.checkpoint_dir, f"test_results_{time.strftime('%Y%m%d-%H%M%S')}.csv")
        df = pd.DataFrame(detailed_results)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"Saved {len(df)} test results to {csv_path}")

        if wandb_logger:
            wandb_logger.log({
                "test_results": wandb.Table(dataframe=df[["true_answer", "generated_response", "is_correct"]])
            })

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay
        )
        total_steps = self.trainer.estimated_stepping_batches
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _get_dataloader(self, split: str) -> DataLoader:
        if split == "train":
            data_dir, shuffle, return_answer = self.hparams.task_data_dir, True, False
        else:
            data_dir, shuffle, return_answer = self.hparams.val_data_dir, False, True

        print(f"Loading {split} dataset from {data_dir}...")
        raw_dataset = KEGGDataset(data_dir=data_dir)
        formatted = [format_kegg_for_dna_llm(raw_dataset[i]) for i in range(len(raw_dataset))]

        processor = DLProcessor(
            tokenizer=self.model.text_tokenizer,
            dna_tokenizer=self.model.dna_tokenizer,
        )
        collate_fn = partial(
            qwen_dna_collate_fn,
            processor=processor,
            max_length_text=self.hparams.max_length_text,
            max_length_dna=self.hparams.max_length_dna,
            return_answer_in_batch=return_answer,
        )
        return DataLoader(
            formatted,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def train_dataloader(self):
        return self._get_dataloader("train")

    def val_dataloader(self):
        return self._get_dataloader("val")

    def test_dataloader(self):
        return self._get_dataloader("test")


# ==============================================================================
# Shared trainer builder
# ==============================================================================

def build_trainer(args, callbacks: list, logger) -> pl.Trainer:
    strategy = (
        "ddp_find_unused_parameters_true"
        if args.strategy == "ddp"
        else DeepSpeedStrategy(
            stage=2, offload_optimizer=False,
            allgather_bucket_size=5e8, reduce_bucket_size=5e8,
        )
    )
    return pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=args.num_gpus,
        strategy=strategy,
        precision="bf16-mixed",
        callbacks=callbacks,
        logger=logger,
        deterministic=False,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=5,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gradient_clip_val=1.0,
        val_check_interval=args.val_check_interval,
    )


# ==============================================================================
# Stage 1 entry point
# ==============================================================================

def main_clip(args):
    """Stage 1: CLIP contrastive pre-training."""
    pl.seed_everything(args.seed)
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    run_name = f"clip-{args.dataset_type}-{args.text_model_name.split('/')[-1]}-{time.strftime('%Y%m%d-%H%M%S')}"
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, run_name)

    model = DNACLIPFineTuner(args)

    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename=run_name + "-{epoch:02d}-{val_loss:.4f}",
            save_top_k=2,
            monitor="val_loss",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        save_dir=args.log_dir,
        name=run_name,
        resume="allow" if args.ckpt_path else None,
    )

    trainer = build_trainer(args, callbacks, logger)
    trainer.fit(model, ckpt_path=args.ckpt_path)
    trainer.validate(model, ckpt_path="best")

    if trainer.global_rank == 0:
        save_dir = os.path.join(args.modelsave_dir, "clip_dna_encoder")
        model.model.dna_model.save_pretrained(save_dir)
        model.dna_tokenizer.save_pretrained(save_dir)
        print(f"CLIP-trained DNA encoder saved to: {save_dir}")


# ==============================================================================
# Stage 2 entry point
# ==============================================================================

def main_llm(args):
    """Stage 2: LLM fine-tuning with CLIP-pretrained DNA encoder."""
    pl.seed_everything(args.seed)
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    run_name = f"llm-{args.text_model_name.split('/')[-1]}-{time.strftime('%Y%m%d-%H%M%S')}"
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, run_name)

    model = DNALLMFineTuner(args)

    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename=run_name + "-{epoch:02d}-{val_loss:.4f}",
            save_top_k=2,
            monitor="val_loss",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]
    logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        save_dir=args.log_dir,
        name=run_name,
        resume="allow" if args.ckpt_path else None,
    )

    trainer = build_trainer(args, callbacks, logger)
    if args.test_only:
        trainer.test(model, ckpt_path=args.ckpt_path or "best")
    else:
        trainer.fit(model, ckpt_path=args.ckpt_path)


# ==============================================================================
# Argument parser
# ==============================================================================

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method("spawn", force=True)

    parser = ArgumentParser(
        description="SeClip: stage 1 = CLIP contrastive pre-training, stage 2 = LLM fine-tuning"
    )

    parser.add_argument("--stage", type=str, choices=["clip", "llm"], required=True,
                        help="'clip' for Stage 1 contrastive pre-training; 'llm' for Stage 2 fine-tuning")

    # Model
    parser.add_argument("--model_type", type=str, choices=["llm", "dna-llm"], default="dna-llm")
    parser.add_argument("--text_model_name", type=str, required=True)
    parser.add_argument("--dna_model_name", type=str, required=True)
    parser.add_argument("--text_model_finetune", type=bool, default=True)
    parser.add_argument("--dna_model_finetune", type=bool, default=False)
    parser.add_argument("--projection_dim", type=int, default=2048,
                        help="CLIP projection dimension (Stage 1 only)")
    parser.add_argument("--offline", type=bool, default=True,
                        help="Load models from local files only (Stage 2)")

    # Training
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_length_dna", type=int, default=1024)
    parser.add_argument("--max_length_text", type=int, default=1024)
    parser.add_argument("--val_check_interval", type=float, default=0.25)
    parser.add_argument("--test_only", action="store_true",
                        help="Skip training; run test evaluation only (Stage 2)")

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Paths
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--modelsave_dir", type=str, default="./saved_models",
                        help="Where to save the DNA encoder after Stage 1")
    parser.add_argument("--clip_dna_model_path", type=str, default=None,
                        help="CLIP checkpoint path for initializing DNA encoder (Stage 2)")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--ckpt_path", type=str, default=None,
                        help="Resume training or evaluate from this checkpoint")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=4)
    parser.add_argument("--strategy", type=str, default="ddp", choices=["ddp", "deepspeed"])

    # Dataset — Stage 1
    parser.add_argument("--dataset_type", type=str, choices=["kegg"], default="kegg")
    parser.add_argument("--kegg_data_dir", type=str, default=None)
    parser.add_argument("--merge_val_test_set", type=bool, default=False)

    # Dataset — Stage 2
    parser.add_argument("--task_data_dir", type=str, default=None,
                        help="Training data directory (Stage 2)")
    parser.add_argument("--val_data_dir", type=str, default=None,
                        help="Validation/test data directory (Stage 2)")

    # Logging
    parser.add_argument("--wandb_project", type=str, default="seclip")
    parser.add_argument("--wandb_entity", type=str, default=None)

    args = parser.parse_args()

    if args.stage == "clip":
        main_clip(args)
    else:
        main_llm(args)
