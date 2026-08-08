
_____________________________________________________________________________________________________________________________________________________________________________

## 🎬AI-Powered Movie Recommendation System:
_____________________________________________________________________________________________________________________________________________________________________________

An intelligent movie recommendation engine built with Python and Machine Learning that suggests films tailored to user preferences. The system analyzes movie metadata (genres, cast, keywords, ratings, etc.) using content-based filtering to generate personalized recommendations, served through an interactive Streamlit web app.

🔗 **Live App:** 

_____________________________________________________________________________________________________________________________________________________________________________

## ✨ Features
_____________________________________________________________________________________________________________________________________________________________________________

- Content-based filtering using ML/NLP techniques (cosine similarity on vectorized features)
- Clean, searchable movie database
- Fast, real-time recommendation generation
- Simple, interactive Streamlit interface

_____________________________________________________________________________________________________________________________________________________________________________

## 🛠️ Tech Stack
_____________________________________________________________________________________________________________________________________________________________________________

- **Language:** Python
- **ML/Data:** scikit-learn, Pandas, NumPy
- **Interface:** Streamlit
- **Deployment:** Streamlit Community Cloud

_____________________________________________________________________________________________________________________________________________________________________________

## 🚀 Getting Started
_____________________________________________________________________________________________________________________________________________________________________________
## Prerequisites:
- Python 3.8+
- pip

### Installation
```bash
git clone https://github.com/sharjeel-shahid-rajpoot-2523/AI-Powered-Movie-Recommendation-System.git
cd AI-Powered-Movie-Recommendation-System
pip install -r requirements.txt
```

### Run locally
```bash
streamlit run app.py
```

> **Note:** The precomputed similarity/movie data (`movie_data.pkl`) is not stored in this repo due to GitHub's 100MB file size limit. It is downloaded automatically at runtime on first launch — no manual setup needed.

_____________________________________________________________________________________________________________________________________________________________________________

## 📊 How It Works
_____________________________________________________________________________________________________________________________________________________________________________

1. Movie dataset is preprocessed and cleaned
2. Features (genres, overview, cast, keywords) are vectorized
3. Cosine similarity scores are computed between movies
4. Top-N most similar movies are returned as recommendations

_____________________________________________________________________________________________________________________________________________________________________________

## 📁 Project Structure
_____________________________________________________________________________________________________________________________________________________________________________

```
├── app.py                
├── Movie_Recomendation_System.ipynb                
├── requirements.txt
├── .gitignore
└── README.md
```

_____________________________________________________________________________________________________________________________________________________________________________

## 🤝 Contributing
_____________________________________________________________________________________________________________________________________________________________________________
Pull requests are welcome. For major changes, please open an issue first to discuss what you'd like to change.
