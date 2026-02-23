# gemma-270m-from-scratch

This project demonstrates full-stack LLM engineering: training from scratch, architecture replication, checkpoint management, and API deployment.

## Project overview
This project implements and trains a 270M parameter decoder-only Transformer from scratch using a Gemma-style architecture.
The model was trained on the TinyStories dataset for autoregressive language modeling and deployed via a FastAPI inference API.
Unlike fine-tuning workflows, this project builds the full training pipeline manually — including architecture definition, optimization loop, and checkpoint management.

| Component | Value |
|:-----------|:------------:|
| **Parameters** | ~270M |
| **Dataset** | TinyStories |
| **Hardware** | 1× NVIDIA T4 (16GB VRAM) |
| **Steps** | 29000 |
| **Batch Size** | 32 |
| **Gradient Accumulation** | 2 |
| **Context Length** | 128 |
| **Validation Loss** | 2.3083 |
| **Throughput** | ~8k tokens/sec |
| **GPU Utilization** | ~100% |
| **VRAM Usage** | ~9GB |

## Architecture Highlights

The model replicates a Gemma-style decoder architecture with:

* RMSNorm (zero-centered scaling)
* Grouped Query Attention (GQA)
* RoPE positional embeddings (local + global)
* Sliding window + full attention layers
* GELU-based gated feed-forward network
* Mixed precision (AMP) training
* Cosine LR schedule with warmup
* Resume-capable checkpointing

## What This Project Demonstrates

This project focuses on full-stack LLM engineering:

* Implementing transformer blocks from scratch
* Handling quadratic attention memory scaling
* Managing VRAM constraints on commodity GPUs
* Correct scheduler placement with gradient accumulation
* Matching architecture exactly during deployment
* Debugging state_dict mismatches
* Building a CPU-compatible inference API

## Inference API

The trained model is wrapped in a FastAPI service.

Run locally:
```uvicorn application.main:app```

Then open:
```http://127.0.0.1:8000/docs```

Example request:
```
{
  "text": "Once upon a time",
  "max_tokens": 80
}
```

## Model Weights

Model weights are not included in this repository due to size constraints.
Available upon request.

If you want to see how i train my model you can see it here
https://www.kaggle.com/code/snehallaldas/gemma-3-270m

It also contains the model weights

