# FAIL: Flow Matching Adversarial Imitation Learning for Image Generation

Official implementation of **Flow Matching Adversarial Imitation Learning (FAIL)** for Image Generation.

FAIL minimizes policy-expert divergence through adversarial training without explicit rewards or pairwise comparisons. We provide two algorithms:
- **FAIL-PD (Pathwise Derivative)**: Backpropagates discriminator gradients through the ODE solver
- **FAIL-PG (Policy Gradient)**: Policy gradient alternative using Flow Policy Optimization (FPO)

Please see [[Paper](https://arxiv.org/abs/xxxx.xxxxx)] for more information.

<table><tr><td>
    Yeyao&nbsp;Ma<sup>1</sup>, Chen&nbsp;Li<sup>2</sup>, Xiaosong&nbsp;Zhang<sup>3</sup>, Han&nbsp;Hu<sup>3</sup>, and Weidi&nbsp;Xie<sup>1</sup>.
    <strong>FAIL: Flow Matching Adversarial Imitation Learning for Image Generation.</strong>
    arXiv, 2026.
</td></tr></table>

<sup>1</sup><em>Shanghai Jiao Tong University</em>, <sup>2</sup><em>Xi'an Jiaotong University</em>, <sup>3</sup><em>Tencent</em>

## Installation

```bash
bash env_setup.sh
```

## Data Preparation

### 1. Download Checkpoints

```bash
mkdir -p ./data/flux ./data/Qwen3-VL-2B-Instruct
```

- [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) → `./data/flux`
- [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) → `./data/Qwen3-VL-2B-Instruct`

### 2. Prepare Expert Data

The expert data consists of:
- `gemini_13k.parquet`: 13K prompts with metadata (uuid, content, etc.)
- Expert images: one image per prompt, organized by uuid

```bash
hf download HansPolo/FAIL-expert-data --repo-type dataset --local-dir ./data
unzip ./data/FAIL_train.zip -d ./data
```

Directory structure after unzip:
```
./data/gemini_13k.parquet
./data/Gemini2K/{uuid}/sample_0.png
```

Each `{uuid}` folder corresponds to a row in `gemini_13k.parquet`, and `sample_0.png` is the expert image for that prompt.

### 3. Preprocess Text Embeddings

Extract FLUX text embeddings for all prompts in the parquet file:

```bash
bash scripts/preprocess/preprocess_flux_rl_embeddings.sh
```

## Training

### Step 1: Cold Start with SFT

First, initialize the policy via Supervised Fine-Tuning on expert demonstrations for one epoch:

```bash
bash scripts/finetune/finetune_flux_sft.sh
```

### Step 2: FAIL Training

Then run FAIL training with the SFT checkpoint (set `--pretrained_transformer_path` in the script):

```bash
# FAIL-PD
bash scripts/finetune/finetune_flux_fail_pd.sh

# FAIL-PG
bash scripts/finetune/finetune_flux_fail_pg.sh
```

Multi-node (e.g., 4 nodes):

```bash
# On each node, set WORLD_SIZE, RANK, MASTER_ADDR
WORLD_SIZE=4 RANK=0 MASTER_ADDR=<master_ip> bash scripts/finetune/finetune_flux_fail_pd.sh  # node 0
WORLD_SIZE=4 RANK=1 MASTER_ADDR=<master_ip> bash scripts/finetune/finetune_flux_fail_pd.sh  # node 1
WORLD_SIZE=4 RANK=2 MASTER_ADDR=<master_ip> bash scripts/finetune/finetune_flux_fail_pd.sh  # node 2
WORLD_SIZE=4 RANK=3 MASTER_ADDR=<master_ip> bash scripts/finetune/finetune_flux_fail_pd.sh  # node 3
```

## Sampling

Generate images using Ray-based distributed inference:

```bash
# Set CHECKPOINT_PATH in the script to load trained model
bash scripts/visualization/sample_flux_ray.sh
```

## Acknowledgement
This repo is built upon these amazing works:
- [FastVideo](https://github.com/hao-ai-lab/FastVideo)
- [DanceGRPO](https://github.com/XueZeyue/DanceGRPO)
- [FPO](https://github.com/akanazawa/fpo)

## Citation

```bibtex
@article{ma2026ail,
  title={FAIL: Flow Matching Adversarial Imitation Learning for Image Generation},
  author={Ma, Yeyao and Li, Chen and Zhang, Xiaosong and Hu, Han and Xie, Weidi},
  journal={arXiv preprint},
  year={2025}
}
```
