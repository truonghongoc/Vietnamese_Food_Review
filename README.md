# 🍜 Food Review NLP

A Vietnamese Natural Language Processing project for **food review sentiment analysis** and **review summarization** using **PhoBERT**, **mT5**, and advanced Vietnamese text preprocessing techniques.

---

## 📌 Project Overview

This project focuses on processing Vietnamese food reviews and solving two main NLP tasks:

* **Sentiment Analysis**

  * Positive
  * Neutral
  * Negative

* **Review Summarization**

  * Generate concise summaries of customer reviews

The project is built using a modular pipeline architecture for maintainability and scalability.

---

## 🚀 Features

* Vietnamese text preprocessing pipeline
* Teencode normalization
* Abbreviation normalization
* Emoji and emoticon normalization
* Negation handling
* Vietnamese word segmentation
* Compound word preservation
* PhoBERT-compatible preprocessing
* Sentiment classification
* Review summarization
* Model evaluation and visualization

---

## 🏗️ Project Structure
See Food_Review_NLP_Project_Structure_Explanation.html for more infomations and explainations (Open in a browser, Chrome for example)

#IMPORTANT#
"test/" folder contains dataset for batch inference demonstrate
"data/" folder contains dataset for model training, validation and testing 
---

## ⚙️ Preprocessing Workflow

```text
Raw Review
      │
      ▼
Unicode Normalization
      │
      ▼
HTML Normalization
      │
      ▼
English Food Normalization
      │
      ▼
Teencode Normalization
      │
      ▼
Abbreviation Normalization
      │
      ▼
Emoji & Emoticon Normalization
      │
      ▼
Negation Normalization
      │
      ▼
URL / Email / Mention Removal
      │
      ▼
Whitespace Normalization
      │
      ▼
Repeated Character Reduction
      │
      ▼
Sentence Tokenization
      │
      ▼
Word Tokenization (underthesea)
      │
      ▼
Negation Joining
      │
      ▼
Compound Word Preservation
      │
      ▼
PhoBERT Format Export
```

---

## 🧠 Models

| Task               | Model       |
| ------------------ | ----------- |
| Sentiment Analysis | PhoBERT     |
|                    |
| Tokenization       | underthesea |

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/your_username/Food_Review_NLP.git
cd Food_Review_NLP
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### 1. Explore Dataset

```bash
jupyter notebook notebooks/01_exploration.ipynb
```

### 2. Preprocess Reviews

```bash
jupyter notebook notebooks/02_preprocessing.ipynb
```

### 3. Train Sentiment Model

```bash
jupyter notebook notebooks/03_sentiment.ipynb
```

## 📝 Example Input

```text
Đồ ăn ngon cực kỳiiii 😍😍😍
Nhân viên phục vụ okeeeeee.
Giá hơi mắc xíu :(
```

---

## 🔧 Preprocessed Output

```text
đồ_ăn ngon cực_kỳ positive_emoji
nhân_viên phục_vụ ok
giá hơi mắc negative_emoji
```

---

## 😊 Sentiment Prediction Example

```text
Input:
"Đồ ăn ngon cực kỳiiii 😍"

Prediction:
Positive

Confidence:
98.73%
```

---



## 📊 Evaluation Metrics

### Sentiment Analysis

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

## 🧪 Testing

Run preprocessing tests:

```bash
python -m pytest tests/
```

---

## 📈 Outputs

Generated results are stored in:

```text
outputs/
├── exploration/
├── figures/
└── metrics/
```

---

## 🛠️ Technologies

* Python
* PyTorch
* Transformers
* PhoBERT
* underthesea
* pandas
* scikit-learn
* matplotlib
* Jupyter Notebook

---

## 🎓 Academic Information

* Course: Natural Language Processing
* Project: Vietnamese Food Review Analysis
* University: Ho Chi Minh City Open University

---

## 📄 License

This project is developed for educational and research purposes.
