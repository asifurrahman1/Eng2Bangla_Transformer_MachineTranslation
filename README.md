# EngToBengaliTranslation

A TensorFlow-based Transformer model for English to Bengali machine translation. This project is modular and extensible—allowing any machine translation task between arbitrary language pairs with appropriate dataset and tokenizer definitions.

---

## 📦 Installation
Clone the repository and install the package using `setup.py`. This will automatically install all dependencies from `requirement.txt`.

```bash
git clone https://github.com/<your-username>/EngToBengaliTranslation.git
cd EngToBengaliTranslation
pip install .
or 
pip install -e

## 🧠 Project Features
✅ Transformer encoder-decoder implementation using TensorFlow 2.x
✅ Modular tokenizer support with BaseTokenizer class
✅ Bengali tokenization using Indic NLP Library
✅ Custom positional encoding and attention masks
✅ Greedy decoding for inference
✅ Easily extendable to other language pairs and datasets
✅ Training and testing entry points via console commands
---
## 🗂️ Directory Structure
EngToBengaliTranslation/
├── Dataset/
│   └── english_to_bangla.csv             # Translation corpus 
├── EngToBengaliTranslation/
│   ├── basetokenizer.py                  # default tokenizer class
│   ├── data_pipeline.py                  # Data preparation as dataloader
│   ├── model.py                          # Transformer architecture
│   ├── model_struct.json                 # Model config (optional use)
│   ├── test.py                           # Evaluation and translation script
│   ├── train.py                          # Model training pipeline
│   └── util.py                           # Utility functions
├── LICENSE                               # MIT License
├── README.md                             # Project documentation
├── requirement.txt                       # Python package dependencies
└── setup.py
---
## 🚀 Getting Started
### Train the Model
Use the console script installed via setup.py:
train-model
Or directly run:
python -m EngToBengaliTranslation.train

### Test / Translate
test-model
Or directly run:
python -m EngToBengaliTranslation.test

## 🔁 Extending to Other Language Pairs
This repository is not limited to English–Bengali. You can adapt it for any language pair by following these steps:
- Prepare a dataset in Dataset, with two columns: source and target.
- Implement a custom tokenizer by subclassing BaseTokenizer:
  class CustomTokenizer(BaseTokenizer):
    def src_tokenizer(self, text):
        # Your custom logic
    def target_tokenizer(self, text):
        # Your custom logic
---
👤 Author
Md Asifur Rahman
Feel free to reach out or open an issue for improvements, collaborations, or research inquiries.



