# Analyzing Human Relational Comprehension in Large Language Audio Models via Conversational Audio

This repository contains code for evaluating Large Audio Language Models (LALMs) on relationship classification tasks using the Meta Seamless Interaction dataset.

## Overview

This project investigates how well state-of-the-art audio language models can identify:
1. **Relationship types** (stranger, friends, family, coworkers, romantic partners, classmates, roommates, neighbors)
2. **Conversation authenticity** (naturalistic vs. improvised/role-play)

## Models Tested

- **AudioFlamingo3** (NVIDIA)
- **Qwen2.5-Omni-7B** (Alibaba)
- **R1-AQA** (MiSpeech)
- **GPT-4o-mini** (OpenAI) - text-based evaluations for creating difficulty-level metric

## Repository Structure

### Data Preparation Scripts

| Script | Description |
|--------|-------------|
| `download_audio.py` | Downloads audio files from Meta Seamless Interaction dataset |
| `merge_and_clean.py` | Combines all Seamless Interaction CSVs into one metadata file |
| `clean_audio_data.py` | Removes rows with missing transcripts and normalizes relationship labels (to be used AFTER audio downloads) |

### Sampling Scripts

| Script | Description |
|--------|-------------|
| `relationship_sample.py` | Stratified random sampling balanced across LLM difficulty and relationship categories |
| `acted_naturalistic_sample.py` | Stratified sampling for naturalistic and improvised conversation samples |

### Model Evaluation Scripts

| Script | Description |
|--------|-------------|
| `test_models.py` | Tests loading of all LALMs to verify installation |
| `llm_easy_hard.py` | Evaluates GPT-5-mini on transcripts to establish text-only baseline |
| `lalm_relationships.py` | Runs LALM predictions for relationship type classification (zero-shot) |
| `lalm_relationships_few_shot.py` | Runs LALM predictions with few-shot examples (8 examples, one per class) |
| `lalm_nat_imp.py` | Classifies audio as naturalistic vs. improvised |
| `acted_relationships.py` | Uses GPT-4o-mini to generate acted relationship descriptors for improvised prompts |

## Setup

### Requirements for LALMs
```bash
pip install -r requirements.txt
```

**Other Script Dependencies:**
- `transformers` (Hugging Face)
- `torch` (PyTorch with CUDA support)
- `librosa` (audio processing)
- `pandas` (data manipulation)
- `openai` (GPT-4o-mini API)
- `scikit-learn` (evaluation metrics)

### Hardware Requirements

- **GPU:** NVIDIA A100 40GB (or equivalent)
- **Storage:** ~500GB for audio files and model cache

## Dataset

**Meta Seamless Interaction Dataset**
- 46,198 audio files
- Naturalistic conversations and improvised role-play scenarios
- 8 relationship categories
- Transcripts available for all conversations

**Corrupted files:** 194 files (44 bytes each - header only, no audio data)

## Usage

### 1. Download and Prepare Data
```bash
# Download audio files
python download_audio.py

# Merge and clean metadata
python merge_and_clean.py
python clean_audio_data.py

# Remove corrupted files
python check_corrupted.py
```

### 2. Create Evaluation Subset
```bash
# Generate stratified sample (1,500 files: 750 naturalistic, 750 improvised)
python acted_naturalistic_sample.py

# Or sample for relationship classification
python relationship_sample.py
```

### 3. Run Model Evaluations
```bash
# Test model loading
python test_models.py --test

# Relationship classification (zero-shot)
python lalm_relationships.py --model all

# Relationship classification (few-shot)
python lalm_relationships_few_shot.py --model all

# Naturalistic vs. Improvised classification
python lalm_nat_imp.py --model all

# Text-only baseline
python llm_easy_hard.py
```

### Run Individual Models
```bash
# Run specific model
python lalm_relationships.py --model AudioFlamingo3
python lalm_relationships.py --model Qwen2.5-Omni
python lalm_relationships.py --model R1-AQA

# Limit number of files for testing
python lalm_relationships.py --model all --limit 10
```

## Output Files

| File | Description |
|------|-------------|
| `audiofiles_transcripts_merged.csv` | Combined metadata from all conversations |
| `lalm_relationships_predictions.csv` | Relationship classification results (zero-shot) |
| `lalm_relationships_predictions_fewshot.csv` | Relationship classification results (few-shot) |
| `lalm_nat_imp_predictions.csv` | Naturalistic vs. improvised classification results |
| `gpt5_predictions_log.txt` | GPT-4o-mini text-only baseline results |
