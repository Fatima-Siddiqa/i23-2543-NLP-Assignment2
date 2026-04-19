# i23-2543 NLP Assignment 2 — Neural NLP Pipeline

**Course:** CS-4063 Natural Language Processing
**Student ID:** i23-2543 | **Section:** DS-A
**Framework:** PyTorch (from scratch)
**GitHub:** https://github.com/Fatima-Siddiqa/i23-2543-NLP-Assignment2

---

## Repository Structure

```
i23-2543-NLP-Assignment2/
├── i23-2543_Assignment2_DS-A/
│   ├── i23-2543_Assignment2_DS-A.ipynb   ← main notebook (all cells executed)
│   ├── embeddings/
│   │   ├── tfidf_matrix.npy              ← Part 1 TF-IDF output
│   │   ├── ppmi_matrix.npy               ← Part 1 PPMI output
│   │   ├── embeddings_w2v.npy            ← Part 2 Skip-gram embeddings (C3, d=100)
│   │   └── word2idx.json                 ← vocabulary mapping
│   ├── models/
│   │   ├── bilstm_pos.pt                 ← trained BiLSTM POS model
│   │   ├── bilstm_ner.pt                 ← trained BiLSTM NER model (with CRF)
│   │   └── transformer_cls.pt            ← trained Transformer classifier
│   └── data/
│       ├── pos_train.conll               ← POS annotated train set
│       ├── pos_test.conll                ← POS annotated test set
│       ├── ner_train.conll               ← NER annotated train set
│       └── ner_test.conll                ← NER annotated test set
└── README.md
```

---

## Prerequisites

- Google Colab account
- The following files uploaded to Google Drive:
  - `cleaned.txt` — preprocessed BBC Urdu corpus
  - `raw.txt` — raw BBC Urdu corpus
  - `Metadata.json` — article metadata

> All trained models, embeddings, and CoNLL data files are already committed to this repository. **You do not need to retrain anything to reproduce all results.**

---

## Quick Start

### Step 1 — Open Notebook in Colab

Upload `i23-2543_Assignment2_DS-A.ipynb` to Google Colab or open directly from Google Drive.

### Step 2 — Run Setup Cells

Run these cells at the top of the notebook:

```python
# Install dependencies
!pip install requests
!apt-get install git-lfs -y -q
!git lfs install
```

```python
# Clone repo — all pretrained files download automatically via Git LFS
import os
from google.colab import userdata

GITHUB_USERNAME = "Fatima-Siddiqa"
GITHUB_TOKEN = userdata.get("GITHUB_TOKEN")  # set in Colab Secrets, or paste directly
REPO_NAME = "i23-2543-NLP-Assignment2"

os.chdir("/content")
!git clone https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git
os.chdir(f"/content/{REPO_NAME}")

base = "i23-2543_Assignment2_DS-A"

# Verify all files are present
!find {base} -type f
```

```python
# Mount Drive (needed for cleaned.txt, raw.txt, Metadata.json)
from google.colab import drive
drive.mount('/content/drive')
```

After cloning, all pretrained files are available locally:
```
embeddings/tfidf_matrix.npy
embeddings/ppmi_matrix.npy
embeddings/embeddings_w2v.npy
embeddings/word2idx.json
models/bilstm_pos.pt
models/bilstm_ner.pt
models/transformer_cls.pt
data/pos_train.conll
data/pos_test.conll
data/ner_train.conll
data/ner_test.conll
```

---

## Reproducing Results Without Retraining

### Part 1 — Load Saved Embeddings

Instead of rerunning Skip-gram training (~25 min), load the saved files directly:

```python
import numpy as np, json

embeddings_w2v = np.load(f"{base}/embeddings/embeddings_w2v.npy")
ppmi_matrix    = np.load(f"{base}/embeddings/ppmi_matrix.npy")
tfidf_matrix   = np.load(f"{base}/embeddings/tfidf_matrix.npy")

with open(f"{base}/embeddings/word2idx.json", encoding="utf-8") as f:
    word2idx = json.load(f)

print(f"embeddings_w2v : {embeddings_w2v.shape}")   # (10000, 100)
print(f"ppmi_matrix    : {ppmi_matrix.shape}")       # (10001, 10001)
print(f"tfidf_matrix   : {tfidf_matrix.shape}")      # (300, 10001)
print(f"Vocabulary     : {len(word2idx)} entries")   # 10000
```

Then run the evaluation cells directly: nearest neighbours, analogy tests, four-condition MRR comparison.

---

### Part 2 — Load Saved BiLSTM Models

Run the dataset preparation cells first (fast, no training), then load saved models:

```python
import torch, numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Run class definition cells in notebook first (BiLSTMTagger, CRF, tag vocabularies)
# Then load:

embeddings_w2v = np.load(f"{base}/embeddings/embeddings_w2v.npy")
VOCAB_SIZE = embeddings_w2v.shape[0]   # 10000
EMBED_DIM  = embeddings_w2v.shape[1]   # 100

model_pos_finetune = BiLSTMTagger(
    vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, hidden_dim=128,
    num_pos_tags=len(POS_TAGS), num_ner_tags=len(NER_TAGS),
    pretrained_embeddings=embeddings_w2v, freeze=False
).to(DEVICE)
model_pos_finetune.load_state_dict(
    torch.load(f"{base}/models/bilstm_pos.pt", map_location=DEVICE)
)
print("POS model loaded!")

model_ner_finetune = BiLSTMTagger(
    vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, hidden_dim=128,
    num_pos_tags=len(POS_TAGS), num_ner_tags=len(NER_TAGS),
    pretrained_embeddings=embeddings_w2v, freeze=False
).to(DEVICE)
model_ner_finetune.load_state_dict(
    torch.load(f"{base}/models/bilstm_ner.pt", map_location=DEVICE)
)
print("NER model loaded!")

# Training F1 values (from original training run)
f1_pos_frozen,   f1_pos_finetune = 0.7440, 0.8395
f1_ner_frozen,   f1_ner_finetune = 0.3614, 0.4629
```

Then run the Section 5 evaluation cells directly: POS accuracy/F1, confusion matrix, NER precision/recall/F1 per entity type, CRF vs no-CRF comparison, ablation study.

---

### Part 3 — Load Saved Transformer Model

Run the dataset preparation cells first, then load the saved model:

```python
# Run Transformer class definition cells in notebook first, then:
transformer.load_state_dict(
    torch.load(f"{base}/models/transformer_cls.pt", map_location=DEVICE)
)
transformer.eval()
print("Transformer model loaded!")
```

Then run the evaluation cells: test accuracy, macro F1, confusion matrix, attention heatmaps.

---

## Expected Results

| Task | Metric | Value |
|---|---|---|
| Skip-gram analogy | Correct / 10 | 9/10 (PASS) |
| Four-condition best | MRR | C1 PPMI = 0.0101 |
| POS tagging | Test Accuracy | 0.9763 |
| POS tagging | Test Macro F1 | 0.8708 |
| NER (with CRF) | Overall F1 | 0.6524 |
| NER (no CRF) | Overall F1 | 0.6463 |
| Transformer | Test Accuracy | 0.5918 |
| Transformer | Macro F1 | 0.2273 |

---

## Notes

- **Git LFS** is configured via `.gitattributes` for `.npy` and `.pt` files. The `git clone` command downloads them automatically — no manual steps required.
- **No retraining needed** — all models are saved and loadable. Use the snippets above to skip training and jump straight to evaluation.
- All notebook cells have been executed and outputs are visible inline in the notebook.
- CoNLL files in `data/` can be inspected directly to verify POS and NER annotations.