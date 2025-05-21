#  Chat Disentanglement via Topic Clustering

In noisy, multi-turn conversations, language models often lose track of context due to rapid topic shifts. This tool **disentangles chat histories into topic-wise threads** using **semantic embeddings + clustering**, enabling more focused LLM behavior in summarization, retrieval, and reasoning tasks.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![NLP](https://img.shields.io/badge/NLP-Topic%20Clustering-orange)
![LLM Ready](https://img.shields.io/badge/LLM-Optimized-critical)

---

## Key Features

-  **Sentence Embedding:** Captures semantic similarity using `all-MiniLM-L6-v2`
-  **Unsupervised Clustering:** Detects coherent topics via Agglomerative Clustering
-  **Thread Segmentation:** Groups utterances into clean topic-wise threads
-  **Optional Topic Labeling:** Uses RAKE to name each thread
-  **LLM-Ready Context Windows:** Improves downstream summarization and QA

---

##  Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/chat-disentanglement.git
cd chat-disentanglement
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download NLTK stopwords (required by RAKE)
```python
import nltk
nltk.download('stopwords')
```

### 4. Run the pipeline
```bash
python main.py
```

---

##  Sample Input (`sample_data/sample_chat.json`)
```json
[
  { "speaker": "user", "message": "Hey, can you help me with my WiFi?" },
  { "speaker": "agent", "message": "Sure, what seems to be the issue?" },
  { "speaker": "user", "message": "Also, I wanted to ask about roaming charges." },
  { "speaker": "agent", "message": "We have international plans for that." },
  { "speaker": "user", "message": "The WiFi cuts out randomly." },
  { "speaker": "agent", "message": "Let me look into that." }
]
```

---

##  Output
```
 Thread 0: wifi issue
------------------------
• Hey, can you help me with my WiFi?
• The WiFi cuts out randomly.
• Let me look into that.

 Thread 1: international plans
-------------------------------
• Also, I wanted to ask about roaming charges.
• We have international plans for that.
```

---

##  Folder Structure

```
chat-disentanglement/
├── main.py
├── requirements.txt
├── README.md
├── sample_data/
│   └── sample_chat.json
└── disentangler/
    ├── embedder.py
    ├── clusterer.py
    ├── labeler.py
    └── utils.py
```

---

##  Use Cases

- Customer support: separate billing vs. tech queries  
- Meeting analysis: extract task items from casual talk  
- Multi-topic chat summarization for LLMs

---
