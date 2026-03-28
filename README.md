# 🔍 Arabic Search Engine with TF-IDF & Sentiment Analysis

## 📌 Overview
This project is an **Arabic Information Retrieval System** built with **TF-IDF weighting** and enhanced with:
- Sentiment Analysis (Positive / Neutral / Negative)
- Boolean, Phrase, and Proximity Search
- Query Expansion using Synonyms
- Interactive UI using Streamlit

It allows users to search Arabic documents efficiently and visualize results with relevance scores and sentiment distribution.

---

## 🚀 Features
- ✅ Arabic text preprocessing (normalization, stopwords removal, stemming)
- ✅ TF-IDF vectorization
- ✅ Cosine similarity ranking
- ✅ Boolean queries (AND, OR, NOT)
- ✅ Phrase search ("...")
- ✅ Proximity search (#k(term1, term2))
- ✅ Sentiment filtering
- ✅ Query expansion (synonyms)
- ✅ Batch query processing
- ✅ Interactive visualization with Plotly

---

## 🧠 Technologies Used
- Python
- Streamlit
- NumPy
- Plotly
- XML processing (ElementTree)

---

## 📂 Project Structure
```
├── app.py                  # Main Streamlit application
├── index_data/             # Saved index files
│   ├── index.pkl
│   ├── index.txt
├── queries.txt             # Batch queries file
├── results.boolean.txt     # Boolean results
├── results.ranked.txt      # Ranked results
├── arabic_stop_words.txt   # Stopwords list
├── twifil.xml              # Dataset
```


## ▶️ Run the Application
```
streamlit run app.py
```

---

## 📊 How It Works

### 1. Indexing
- Load XML dataset
- Preprocess Arabic text
- Build inverted index
- Compute TF-IDF weights

### 2. Searching
- Convert query into TF-IDF vector
- Retrieve candidate documents
- Rank using cosine similarity

### 3. Sentiment Analysis
- Extract sentiment from XML
- Filter or analyze result distribution

---

## 🔎 Example Queries

- Simple search:
```
مشكلة
```

- Phrase search:
```
"التعليم في الجزائر"
```

- Boolean search:
```
تعليم AND اقتصاد
```

- Proximity search:
```
#3(تعليم, اقتصاد)
```

---

## 📈 Output
- Ranked results with scores
- Highlighted query terms
- Sentiment distribution (pie chart)
- Downloadable result files

---

## 👩‍💻 Author
**Lamis Nesrine Abbou**

---

## 💡 Future Improvements
- Add deep learning ranking models (BERT)
- Improve Arabic stemming (use advanced NLP libraries)
- Deploy as a web service (FastAPI + Docker)

---

## 📜 License
This project is for academic and educational purposes.
