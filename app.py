import streamlit as st
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from transformers import pipeline


# téléchargement des ressources nécessaires
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# initialisation
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


# MODÈLE D'ANALYSE UTILISÉ
roberta_sentiment = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment"
)

# RoBERTa
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
        return 'neutral'  # fallback en cas d'erreur'''



# MAIN AFFICHAGE
def analyse_sentiments(dataframes, labels): 
    label_to_code = {v: k for k, v in labels.items()}
    st.title("Analyse des Sentiments 🎭")

    # chargement de dataframe
    if "tweets_with_sentiments" not in dataframes:
        st.error("tweets_with_sentiments est manquant dans les données chargées")
        return

    df = pd.read_csv('tweets_with_sentiments.csv')

    crises = df["topic"].dropna().unique()
    crises_lisibles = [labels.get(code, code) for code in sorted(crises)]
    selected_label = st.selectbox("Choisissez une crise 📍 ", crises_lisibles)
    selected_crisis = label_to_code.get(selected_label, selected_label)

    # filtrage des données pour la crise sélectionnée
    df = df[df['topic'] == selected_crisis]

    # dictionnaire de couleurs pour les sentiments
    set3_colors = px.colors.qualitative.Set3
    color_map = {
        'positive': set3_colors[1],
        'neutral': set3_colors[0],
        'negative': set3_colors[2]
    }



    # RÉPARTITION DES SENTIMENTS
    st.subheader("Répartition des sentiments 📊")

    # Barplot
    roberta_counts = df['roberta_label'].value_counts().reset_index()
    roberta_counts.columns = ['Sentiment', 'Tweets']
    roberta_counts['Color'] = roberta_counts['Sentiment'].map(color_map)

    fig_bar = px.bar(roberta_counts, 
                        x='Sentiment', y='Tweets',
                        title=None,
                        color='Sentiment', color_discrete_map = color_map)
    fig_bar.update_layout(barmode='stack')


    # Pie Chart
    col = 'roberta_label'
    title = ''

    value_counts = df[col].value_counts()
    labels = value_counts.index.tolist()
    values = value_counts.values.tolist()

    donut_colors = [color_map[label] for label in labels]

    fig_pie = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,  # Donut style
        marker=dict(colors=donut_colors),
        textinfo='percent',
    )])

    fig_pie.update_layout(
        title=title,
        margin=dict(t=50, b=0, l=0, r=0),
        showlegend=False
    )

    # affichage
    col1, col2 = st.columns([2,1])
    with col1:
        st.plotly_chart(fig_bar, use_container_width=True)
    with col2:
        st.plotly_chart(fig_pie, use_container_width=True)



    # ÉVOLUTION DANS LE TEMPS
    st.subheader("Évolution des sentiments dans le temps ⏳")

    # vérification que les dates sont bien converties
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df['date'] = df['created_at'].dt.date

    # mapper les labels en scores numériques
    sentiment_map = {'negative': -1, 'neutral': 0, 'positive': 1}
    df['vader_score'] = df['vader_label'].map(sentiment_map)

    # moyenne quotidienne des scores
    daily_sentiment = df.groupby('date')['vader_score'].mean().reset_index()

    # affichage
    fig = px.line(
        daily_sentiment,
        x='date',
        y='vader_score',
        title='Average Daily Sentiment',
        markers=True,
        labels={'vader_score': 'Average Sentiment (-1=neg, 1=pos)', 'date': 'Date'},
    )

    fig.update_layout(
        xaxis_title='Date',
        yaxis_title='Average Sentiment',
        xaxis_tickangle=-45,
        yaxis=dict(dtick=0.5),
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True)


















