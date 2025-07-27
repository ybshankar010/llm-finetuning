# Fine-tuning LLMs for Sentiment Analysis: DistilBERT vs Qwen3-4B

A comprehensive comparison of fine-tuning approaches for sentiment analysis using transformer models and XGBoost, demonstrating memory-efficient techniques and practical implementations.

## 🎯 Overview

This repository explores the effectiveness of fine-tuning different sized language models for sentiment analysis tasks. We compare:

- **DistilBERT** (67M parameters) - Compact and efficient
- **Qwen3-4B** (4B parameters) - Large and capable

Using a hybrid approach: **Transformer Embeddings + XGBoost Classification**

## 📊 Key Results

| Model | Configuration | Validation Accuracy | Test Accuracy | Improvement |
|-------|---------------|-------------------|---------------|------------|
| **DistilBERT** | Pre-trained + XGBoost | 82.30% | 82.62% | Baseline |
| **DistilBERT** | Fine-tuned + XGBoost | **91.20%** | **89.84%** | **+7.22%** |
| **Qwen3-4B** | Pre-trained + XGBoost | 89.00% | 90.02% | Baseline |
| **Qwen3-4B** | Fine-tuned + XGBoost | **93.40%** | **93.18%** | **+3.16%** |

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch transformers scikit-learn xgboost tqdm numpy pandas peft bitsandbytes accelerate
```

### Basic Usage

```python
# Load and prepare data
from data_utils import prepare_data_smart
X_train, y_train, X_val, y_val, X_test, y_test = prepare_data_smart()

# Clean GPU memory
from common_utils import cleanup_gpu_memory
cleanup_gpu_memory()

# Extract baseline embeddings
from embedding_utils import get_embeddings
embeddings = get_embeddings(texts, model, tokenizer, device, "baseline")

# Train classifier
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1)
xgb.fit(embeddings, labels)
```

## 🔧 Core Components

### 1. Memory-Efficient Model Loading

```python
from transformers import BitsAndBytesConfig

# 4-bit quantization for memory efficiency
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
```

### 2. Embedding Extraction

```python
def get_embeddings(texts, model, tokenizer, device, model_type="base"):
    """Extract rich semantic embeddings from transformer models"""
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), 8)):
            batch = texts[i:i+8]
            enc = tokenizer(batch, return_tensors="pt", padding=True, 
                          truncation=True, max_length=512)
            enc = {k: v.to(device) for k, v in enc.items()}
            
            outputs = model(**enc)
            # Mean pooling for sentence-level representations
            embeddings_batch = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(embeddings_batch.cpu().numpy())
    
    return np.vstack(embeddings)
```

### 3. LoRA Fine-tuning for Large Models

```python
from peft import LoraConfig, get_peft_model

# Efficient fine-tuning with LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,
    lora_alpha=64,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "o_proj", "down_proj"],
)

model = get_peft_model(base_model, lora_config)
# Only trains ~1.7% of parameters!
```

## 🔬 Experiments

### DistilBERT Pipeline

1. **Baseline**: Pre-trained DistilBERT → Embeddings → XGBoost
2. **Fine-tuned**: Fine-tuned DistilBERT → Embeddings → XGBoost

```bash
# Run DistilBERT experiment
python -m notebooks.distilbert_experiment
```

### Qwen3-4B Pipeline

1. **Baseline**: Pre-trained Qwen3-4B → Embeddings → XGBoost  
2. **Fine-tuned**: LoRA Fine-tuned Qwen3-4B → Embeddings → XGBoost

```bash
# Run Qwen experiment (requires ~8GB GPU memory)
python -m notebooks.qwen_experiment
```

## 💾 Memory Management

This repository includes comprehensive GPU memory management:

```python
from common_utils import cleanup_gpu_memory

# Essential for switching between models
cleanup_gpu_memory()
```

**Memory Requirements:**
- **DistilBERT**: ~2GB GPU memory (with quantization)
- **Qwen3-4B**: ~3GB GPU memory (with 4-bit quantization + LoRA)

## 📈 Key Insights

### 1. **Fine-tuning Impact Varies by Model Size**
- **DistilBERT**: +7.22% improvement (huge impact)
- **Qwen3-4B**: +3.16% improvement (solid but smaller)

### 2. **Larger Models Have Better Baselines**
- Qwen3-4B starts at 90% accuracy vs DistilBERT's 82%
- Shows the power of scale and modern training

### 3. **Hybrid Approach Benefits**
- More stable than end-to-end fine-tuning
- Easier to debug and iterate
- Memory efficient
- Robust classification with XGBoost

### 4. **Practical Trade-offs**
- **Limited resources**: DistilBERT + fine-tuning (great value)
- **Best performance**: Qwen3-4B + LoRA (worth the cost)
- **Prototyping**: Start small, scale up as needed

## 🛠️ Advanced Usage

### Custom Dataset

```python
# Prepare your own dataset
from data_utils import prepare_and_save_data

data = prepare_and_save_data('pickle')  # Saves for reuse
X_train, y_train, X_val, y_val, X_test, y_test = load_data()
```

### Model Comparison

```python
# Compare multiple models easily
from model_utils import compare_models

results = compare_models(
    models=['distilbert', 'qwen3-4b'],
    datasets=[X_train, X_val, X_test],
    labels=[y_train, y_val, y_test]
)
```

### Memory-Efficient Fine-tuning

```python
# For large models with limited GPU
from model_utils import load_finetuned_model_memory_efficient

model, tokenizer = load_finetuned_model_memory_efficient(
    model_path="./saved_models/qwen_finetuned"
)
```

## 📋 Requirements

### Hardware
- **Minimum**: 8GB GPU memory (for DistilBERT)
- **Recommended**: 12GB+ GPU memory (for Qwen3-4B)
- **CPU**: 16GB+ RAM recommended

### Software
- Python 3.8+
- PyTorch 2.0+
- transformers 4.30+
- CUDA 11.8+ (for GPU acceleration)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Areas for Contribution
- Additional model comparisons
- More efficient memory management
- Alternative embedding strategies
- Extended evaluation metrics

## 📄 License

This project is licensed under the APACHE License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Hugging Face for the transformers library
- Alibaba for the Qwen3 model
- The open-source community for tools and datasets

## 📖 Related Work

- [Fine-tuning LLMs for Sentiment Analysis Article](link-to-medium-article)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)
- [Qwen Technical Report](https://arxiv.org/abs/2309.16609)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)

## 🔗 Citation

If you use this work in your research, please cite:

```bibtex
@misc{llm_sentiment_comparison_2025,
  title={Fine-tuning LLMs for Sentiment Analysis: A Practical Comparison},
  author={Bhavani Shankar Y},
  year={2025},
  url={https://github.com/yourusername/llm-sentiment-comparison}
}
```

---

**Star ⭐ this repo if you found it helpful!**
