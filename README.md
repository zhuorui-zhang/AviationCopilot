# AviationCopilot ✈️

This repository contains the data and code for training AviationCopilot through knowlegde-structure-aware continual pretraining and supervised fine-tuning, as described in our research paper.

## 📋 Table of Contents

- Overview
- Project Structure
- Dataset Description
- Installation
- Training Pipeline
  - Continual Pretraining
  - Supervised Fine-tuning
  - Merging PEFT Adapters
- Usage Examples
- Acknowledgments
- Citation

## 🎯 Overview

This project focuses on adapting large language models to the aviation domain through a two-stage training process:

1. **Continual Pretraining (CPT)**: Domain adaptation using aviation-specific corpora
2. **Supervised Fine-tuning (SFT)**: Task-specific training for instruction following

The training pipeline supports:
- Full parameter fine-tuning and parameter-efficient methods (LoRA/QLoRA)
- Multi-GPU distributed training
- Flexible data loading from multiple sources
- Gradient checkpointing for memory efficiency

## 📁 Project Structure

```
.
├── data/
│   ├── CPT/                          # Continual pretraining data
│   │   ├── aviation_wiki_main.jsonl
│   │   ├── aviation_wiki_sub_important.jsonl
│   │   ├── knowledge_structure_absorb.jsonl
│   │   ├── all_txt_filtered_chunks_datajuicer.jsonl
│   │   ├── all_txt_filtered_chunks_datajuicer_instruction_augmented_M-3-shot.jsonl
│   │   ├── fineweb_sample_10k.jsonl
│   │   ├── c4_sample_part.jsonl
│   │   ├── cc_2023-06_sample_part.jsonl
│   │   └── stackexchange_sample_part.jsonl
│   ├── SFT/                          # Supervised fine-tuning data
│   ├── RAW/                          # Raw aviation documents
│   └── evaluation/                   # Evaluation datasets
├── pretraining.py                    # Continual pretraining script
├── supervised_finetuning.py          # Supervised fine-tuning script
├── merge_peft_adapter.py             # PEFT adapter merging utility
├── run_pt.sh                         # Pretraining execution script
├── run_sft.sh                        # Fine-tuning execution script
├── run_merge.sh                      # Adapter merging script
└── requirements.txt                  # Python dependencies
```

## 📊 Dataset Description

### Continual Pretraining Data (CPT)

Located in data/CPT, this directory contains domain-specific corpora:

**Aviation Knowledge Base**:
- aviation_wiki_main.jsonl: Core aviation encyclopedia content
- aviation_wiki_sub_important.jsonl: Supplementary aviation topics
- knowledge_structure_absorb.jsonl: Structured aviation knowledge

**Technical Documentation**:
- all_txt_filtered_chunks_datajuicer.jsonl: Processed aviation technical documents from all_txt
- all_txt_filtered_chunks_datajuicer_instruction_augmented_M-3-shot.jsonl: Instruction-augmented version

**General Corpora Samples** (for continual learning stability):
- fineweb_sample_10k.jsonl: High-quality web text
- c4_sample_part.jsonl: C4 dataset samples
- cc_2023-06_sample_part.jsonl: Common Crawl samples
- stackexchange_sample_part.jsonl: Technical Q&A data

### Supervised Fine-tuning Data (SFT)

Located in data/SFT, containing instruction-following datasets for aviation-specific tasks.

### Raw Data

contains unprocessed aviation technical documents including:
- FAA regulations and guidelines
- Safety risk management documents
- Air quality analysis handbooks
- Aviation technical support documents

## 🔧 Installation

```bash
# Clone the repository
git clone <repository-url>
cd <repository-name>

# Install dependencies
pip install -r requirements.txt
```

### Requirements

Key dependencies include:
- PyTorch >= 2.0
- Transformers >= 4.30.0
- Datasets
- PEFT (Parameter-Efficient Fine-Tuning)
- DeepSpeed (optional, for large-scale training)
- Loguru

## 🚀 Training Pipeline

### Continual Pretraining

Continual pretraining adapts the base model to aviation domain using pretraining.py.

**Key Features**:
- Supports both `.txt` and `.jsonl` formats
- Automatic text chunking with configurable block_size
- Optional text grouping for efficient training
- Mixed precision training (fp16/bf16)
- LoRA/QLoRA support for memory efficiency

**Example Command**:

```bash
bash run_pt.sh
```

**Configuration** (see run_pt.sh):

```bash
torchrun --nproc_per_node 4 pretraining.py \
    --model_name_or_path <base-model-path> \
    --train_file_dir ./data/CPT \
    --validation_file_dir ./data/CPT \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --use_peft True \
    --block_size 1024 \
    --num_train_epochs 3 \
    --learning_rate 2e-5 \
    --output_dir ./output/cpt_model
```

**Key Arguments** (defined in DataArguments):
- `--train_file_dir`: Directory containing training files (.txt/.jsonl)
- `--validation_file_dir`: Directory containing validation files
- `--block_size`: Sequence length for training (default: 1024)
- `--max_train_samples`: Limit training samples for debugging
- `--preprocessing_num_workers`: Parallel workers for data preprocessing

### Supervised Fine-tuning

Fine-tune the pretrained model for instruction-following using supervised_finetuning.py

**Example Command**:

```bash
bash run_sft.sh
```

**Configuration** (see run_sft.sh):

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node 1 supervised_finetuning.py \
    --model_type auto \
    --model_name_or_path <pretrained-model-path> \
    --train_file_dir ./data/SFT \
    --validation_file_dir ./data/SFT \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 24 \
    --do_train \
    --do_eval \
    --template_name qwen \
    --use_peft True \
    --max_train_samples -1 \
    --max_eval_samples 10 \
    --model_max_length 256 \
    --num_train_epochs 3 \
    --learning_rate 4e-5 \
    --output_dir ./output/sft_model
```

**Data Format**:

The SFT data should be in JSONL format with instruction format:

```json
{
  "instruction": "Explain aviation safety procedures",
  "input": "In the context of commercial flights",
  "output": "Aviation safety procedures include...",
  "answer": ""
}
```

### Merging PEFT Adapters

After training with LoRA/QLoRA, merge adapters back to base model using 

merge_peft_adapter.py:

```bash
bash run_merge.sh
```

## 💡 Usage Examples

### 1. Full Pipeline Training

```bash
# Step 1: Continual Pretraining
bash run_pt.sh

# Step 2: Merge adapters (if using PEFT)
bash run_merge.sh #(you should replace the lora_model path with the one you obtained in step1)

# Step 3: Supervised Fine-tuning
bash run_sft.sh

# Step 4: Merge adapters (if using PEFT)
bash run_merge.sh #(you should replace the lora_model path with the one you obtained in step3)
```

### 2. Custom Data Training

```bash
# Organize your data
mkdir -p custom_data/CPT custom_data/SFT

# Place your .jsonl or .txt files in respective folders
# Then modify the training scripts:

python pretraining.py \
    --train_file_dir ./custom_data/CPT \
    --model_name_or_path <model-path> \
    ...
```

### 3. Memory-Efficient Training

For limited GPU memory, use QLoRA with gradient checkpointing:

```bash
python supervised_finetuning.py \
    --use_peft True \
    --load_in_4bit True \
    --gradient_checkpointing True \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    ...
```

## 🙏 Acknowledgments

This project builds upon several excellent open-source projects:

- **[Hugging Face Transformers](https://github.com/huggingface/transformers)**: Core framework for model training and inference
- **[PEFT](https://github.com/huggingface/peft)**: Parameter-efficient fine-tuning methods (LoRA/QLoRA)
- **[DeepSpeed](https://github.com/microsoft/DeepSpeed)**: Distributed training optimization
- **[MinerU](https://github.com/opendatalab/MinerU)**: Dataset Preparation
- **[MinerU](https://github.com/microsoft/LMOps)**: Dataset Preparation
- **[MedicalGPT](https://github.com/shibing624/MedicalGPT)**: Training pipeline inspiration, We highly recommend checking out the original repository for comprehensive training examples and best practices. Our implementation is adapted from their excellent work.
- **[TigerBot](https://github.com/TigerResearch/TigerBot)**: Training pipeline inspiration, We highly recommend checking out the original repository for comprehensive training examples and best practices. Our implementation is adapted from their excellent work.

We also acknowledge the aviation regulatory bodies (FAA) and oneline learning website (<https://flightapprentice.com/>, <https://www.aviationseminars.com/>) whose technical documentation forms the foundation of our domain-specific corpus.

## 📝 Citation
If you find our work helpful, please consider citing our paper:
```bibtex
@article{zhang2026aviationcopilot,
  title={AviationCopilot: Building a reliable LLM-based Aviation Copilot inspired by human pilot training},
  author={Zhang, Zhuorui and Feng, Shanshan and Yang, Tiance and Huang, Ruobing and Wang, Hao and Wang, Fu and Li, Fan},
  journal={Advanced Engineering Informatics},
  volume={69},
  pages={103806},
  year={2026},
  publisher={Elsevier}
}
```
