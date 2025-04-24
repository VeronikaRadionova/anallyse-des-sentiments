import streamlit as st
import pandas as pd
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Télécharger les ressources nécessaires
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')

# Initialisation
lemmatizer = WordNetLemmatizer()

def clean_text_for_sentiment(text):
    text = str(text).lower()

    # earthquake magnitudes ("m 4.5" → "magnitude_4_5")
    text = re.sub(r'(m|magnitude)\s*[\:\.]?\s*([0-9]+\.[0-9]+|[0-9]+)',
                 lambda m: f"magnitude_{m.group(2).replace('.', '_')}",
                 text, flags=re.IGNORECASE)

    # dates/timestamps
    text = re.sub(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{1,2},?\s*\d{4}\s*\d{1,2}:\d{2}(:\d{2})?\s*(gmt|utc)?\b',
                  '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', '', text)

    # URLs, hashtags, mentions, punctuation
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\brt\b', '', text)
    text = re.sub(r'\bvia\b', '', text)
    text = re.sub(r'[^\w\s]|(?<!\w)(?=\d)|(?<=\d)(?!\w)', ' ', text)

    # tokenizer
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [lemmatizer.lemmatize(w) for w in words
             if w not in stop_words and len(w) > 1]

    return ' '.join(words)



# Chargement du fichier CSV via Streamlit
st.title("Analyse des Sentiments des Tweets")

# Télécharger un fichier
uploaded_file = st.file_uploader("Choisir un fichier CSV", type=["csv"])

if uploaded_file is not None:
    # Lire le fichier CSV
    df = pd.read_csv(uploaded_file)
    
    # Vérifier que le fichier contient une colonne 'text'
    if 'text' in df.columns:
        # Nettoyage des tweets
        df['clean_text'] = df['text'].apply(clean_text_for_sentiment)
        
        # Afficher les premiers résultats pour validation
        st.subheader("Aperçu des données nettoyées")
        st.write(df[['text', 'clean_text']].head())
        
        # Enregistrer le fichier nettoyé
        df.to_csv('tweets_cleaned_text.csv', index=False)
        
        # Afficher les premiers tweets nettoyés
        st.subheader("Tweets nettoyés")
        st.write(df['clean_text'].iloc[1:10])
        
        # Créer un graphique Plotly (ex: nombre de tweets par longueur de texte)
        df['text_length'] = df['clean_text'].apply(len)
        fig = px.histogram(df, x='text_length', nbins=30, title="Distribution de la longueur des tweets nettoyés")
        
        st.plotly_chart(fig)
    else:
        st.error("Le fichier CSV doit contenir une colonne 'text'.")