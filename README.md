# LLM Project

## Project Task
I created a **sentiment analysis** classifier that predicts whether a given IMDB movie review is **positive** or **negative**. This helps summarize audience opinions at scale—useful for things like app-store dashboards or recommendation engines.

## Dataset
- **Source**: [IMDB movie reviews](https://huggingface.co/datasets/imdb) (Hugging Face)  
- **Size**: 50 000 reviews (25 000 positive / 25 000 negative)  
- **Splits**:  
  - **Train**: 22 500  
  - **Validation**: 2 500  
  - **Test**: 25 000  
- **Format**: Plain-text reviews with binary sentiment labels

## Pre-trained Model
I selected **DistilBERT** (`distilbert-base-uncased`) as the backbone for fine-tuning:

- **Architecture**: 6 Transformer layers (vs. 12 in BERT-base), 768 hidden units, 12 heads  
- **Parameters**: ~66 million (≈40 % smaller than BERT-base)  
- **Pre-training**: distilled from BERT-base on Wikipedia + BookCorpus  
- **Tokenizer**: uncased WordPiece (30 522-token vocabulary)  
- **Fine-tuned Head**: SST-2 sentiment classifier checkpoint from Hugging Face  
- **Why**: strikes a great balance of accuracy, speed, and resource efficiency for binary text classification  

## Performance Metrics

I tracked **Accuracy** and **F1** on the held-out test split:


| Variant                                     | Epochs |   LR   |   WD   |    Warmup    | Grad Accum | Test Acc | Test F1 |
|---------------------------------------------|:------:|:------:|:------:|:------------:|:----------:|:--------:|:-------:|
| **Baseline** (`distilbert-base-uncased-finetuned`) |   3    |  2e-5  |  0.01  | None         |     1      | **0.9119** | **0.9103** |
| **Optimized v1**                            |   4    |  3e-5  |   0    | 500 steps    |     1      |   0.9066 |   0.9080 |
| **Optimized v2**                            |   3    |  2e-5  |  0.01  | 10 % steps   |     2      |   0.9091 |   0.9100 |
| **Optimized v3**                            |   4    |  1e-5  |  0.02  | 20 % steps   |     2      |   0.9090 |   0.9086 |

> **Best result**: the **baseline** run still achieved top test accuracy (91.19 %).
> Although v1–v3 explored higher learning rates, additional epochs, warmup steps, and gradient accumulation, these changes either led to under- or over-regularization, slower convergence, or slight over-fitting—so the original hyperparameters struck the best balance on the dataset.


## Hyperparameters
I found the following “knobs” to be the most impactful when tuning my DistilBERT sentiment classifier:

- **Learning Rate (LR)**  
  Controls how quickly the model updates its weights. Too high (3e-5) led to rapid but unstable learning and over-fitting; too low (1e-5) was stable but slow; 2e-5 struck the best balance.

- **Number of Epochs**  
  More epochs let the model learn longer, but beyond 3 epochs it began to over-fit on our training split.

- **Weight Decay (WD)**  
  A small value (0.01) helped reduce over-fitting without hindering learning; removing it entirely led to under-regularization.

- **Warmup Steps / Ratio**  
  Ramping up the LR for the first few hundred steps or 10–20 % of training stabilized the early gradients and improved convergence.

- **Gradient Accumulation**  
  Accumulating over 2 steps gave an effective batch size of 32 (on limited GPU memory), smoothing gradient estimates at the cost of slower per-update throughput.

Below is a quick summary of what I actually tried and observed:

| Hyperparameter      | Values Tried     | Observation                                                |
|---------------------|------------------|------------------------------------------------------------|
| **Learning Rate**   | 1e-5, 2e-5, 3e-5 | 2e-5 was most stable; 3e-5 over-fit; 1e-5 converged slowly. |
| **Epochs**          | 3, 4             | 3 epochs balanced learning vs. over-fit best.              |
| **Weight Decay**    | 0, 0.01, 0.02    | 0.01 reduced over-fitting; 0 (no decay) under-regularized. |
| **Warmup**          | None, 500 steps, 20 % steps | Brief warmup improved early training stability. |
| **Grad Accum**      | 1, 2             | 2× accumulation smoothed updates (bs=32) on a small GPU.   |32.                     |

## 🔗 Relevant Links

- **My model on Hugging Face**:  
  https://huggingface.co/rxnu/imdb-distilbert-finetuned  
- **Dataset on Hugging Face**:  
  https://huggingface.co/datasets/imdb  


