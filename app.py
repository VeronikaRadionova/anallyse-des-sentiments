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
nltk.download('punkt_tab')

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




from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# TextBlob
def analyze_textblob(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.05:
        label = 'positive'
    elif polarity < -0.05:
        label = 'negative'
    else:
        label = 'neutral'
    return polarity, label

# VADER
vader = SentimentIntensityAnalyzer()

def analyze_vader(text):
    scores = vader.polarity_scores(text)
    compound = scores['compound']
    if compound > 0.05:
        label = 'positive'
    elif compound < -0.05:
        label = 'negative'
    else:
        label = 'neutral'
    return compound, label



# Chargement du fichier CSV via Streamlit
st.title("Analyse des Sentiments des Tweets")

df = pd.read_csv('Tweet_clean.csv')
df['clean_text'] = df['text'].apply(clean_text_for_sentiment)




# Affichage des résultats
#st.subheader("Aperçu des données nettoyées")
#st.write(df[['text', 'clean_text']].head())

# Enregistrer le fichier nettoyé
#df.to_csv('tweets_cleaned_text.csv', index=False)

# Affichage des premiers tweets nettoyés
#st.subheader("Tweets nettoyés")
#st.write(df['clean_text'].iloc[1:10])





# Analyse TextBlob
df[['textblob_polarity', 'textblob_label']] = df['clean_text'].apply(
    lambda x: pd.Series(analyze_textblob(x))
)

# Analyse VADER
df[['vader_compound', 'vader_label']] = df['clean_text'].apply(
    lambda x: pd.Series(analyze_vader(x))
)




# Affichage des résultats dans Streamlit
#st.subheader("Aperçu des données analysées")
#st.write(df[['text', 'clean_text', 'textblob_polarity', 'textblob_label', 'vader_compound', 'vader_label']].head())

# Enregistrer le fichier nettoyé avec les analyses de sentiment
#df.to_csv('tweets_with_sentiments.csv', index=False)

# Affichage des premiers tweets avec leurs sentiments
#st.subheader("Tweets avec Sentiments")
#st.write(df[['text', 'textblob_label', 'vader_label']].head(10))





from transformers import pipeline

roberta_sentiment = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)

# Fonction pour obtenir le label de sentiment avec RoBERTa
def get_roberta_label(text):
    try:
        result = roberta_sentiment(text)
        label = result[0]['label'].lower()
        if label == 'label_0':
            return 'negative'
        elif label == 'label_1':
            return 'neutral'
        elif label == 'label_2':
            return 'positive'
        else:
            return 'unknown'
    except:
        return 'neutral'  # fallback en cas d'erreur
    
# Analyse RoBERTa
df['roberta_label'] = df['clean_text'].apply(get_roberta_label)


'''# Affichage des résultats dans Streamlit
st.subheader("Aperçu des données analysées")
st.write(df[['text', 'clean_text', 'textblob_polarity', 'textblob_label', 
            'vader_compound', 'vader_label', 'roberta_label']].head())

# Enregistrer le fichier nettoyé avec les analyses de sentiment
df.to_csv('tweets_with_sentiments.csv', index=False)

# Affichage des premiers tweets avec leurs sentiments
st.subheader("Tweets avec Sentiments")
st.write(df[['text', 'textblob_label', 'vader_label', 'roberta_label']].head(10))
'''




import plotly.express as px

# Création des graphiques interactifs avec Plotly

# Graphique pour TextBlob
fig_textblob = px.bar(df['textblob_label'].value_counts().reset_index(), 
                      x='index', y='textblob_label',
                      title="TextBlob Sentiment Distribution",
                      labels={'index': 'Sentiment', 'textblob_label': 'Count'},
                      color='index', color_discrete_sequence=px.colors.qualitative.Set2)
fig_textblob.update_layout(barmode='stack')

# Graphique pour VADER
fig_vader = px.bar(df['vader_label'].value_counts().reset_index(), 
                   x='index', y='vader_label',
                   title="VADER Sentiment Distribution",
                   labels={'index': 'Sentiment', 'vader_label': 'Count'},
                   color='index', color_discrete_sequence=px.colors.qualitative.Set2)
fig_vader.update_layout(barmode='stack')

'''# Graphique pour RoBERTa
fig_roberta = px.bar(df['roberta_label'].value_counts().reset_index(), 
                     x='index', y='roberta_label',
                     title="RoBERTa Sentiment Distribution",
                     labels={'index': 'Sentiment', 'roberta_label': 'Count'},
                     color='index', color_discrete_sequence=px.colors.qualitative.Set2)
fig_roberta.update_layout(barmode='stack')'''

# Affichage des graphiques dans Streamlit
st.subheader("Répartition des sentiments (TextBlob, VADER, RoBERTa)")
st.plotly_chart(fig_textblob)
st.plotly_chart(fig_vader)
#st.plotly_chart(fig_roberta)

