import streamlit as st
import pandas as pd
import re
import plotly.express as px
import plotly.graph_objects as go

'''import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


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




from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline

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
        return 'neutral'  # fallback en cas d'erreur


# chargement du fichier CSV
df = pd.read_csv('Tweet_clean.csv')
df['clean_text'] = df['text'].apply(clean_text_for_sentiment)


# analyse TextBlob
df[['textblob_polarity', 'textblob_label']] = df['clean_text'].apply(
    lambda x: pd.Series(analyze_textblob(x))
)

# analyse VADER
df[['vader_compound', 'vader_label']] = df['clean_text'].apply(
    lambda x: pd.Series(analyze_vader(x))
)

# analyse RoBERTa
#df['roberta_label'] = df['clean_text'].apply(get_roberta_label)
'''








# MAIN AFFICHAGE

#def analyse_sentiments(dataframes, labels): 
st.title("Analyse des Sentiments des Tweets")

'''if "tweets_with_sentiments" not in dataframes:
        st.error("tweets_with_sentiments est manquant dans les données chargées")
        return'''

df = pd.read_csv('tweets_with_sentiments.csv')

# dictionnaire de couleurs pour les sentiments
set3_colors = px.colors.qualitative.Set3

color_map = {
    'positive': set3_colors[1],
    'neutral': set3_colors[0],
    'negative': set3_colors[2]
}

# RoBERTa
roberta_counts = df['roberta_label'].value_counts().reset_index()
roberta_counts.columns = ['Sentiment', 'Tweets']
roberta_counts['Color'] = roberta_counts['Sentiment'].map(color_map)

fig_roberta = px.bar(roberta_counts, 
                        x='Sentiment', y='Tweets',
                        title=None,
                        color='Sentiment', color_discrete_map = color_map)
fig_roberta.update_layout(barmode='stack')

# affichage
st.subheader("Répartition des sentiments RoBERTa")
#st.plotly_chart(fig_roberta)


#st.subheader("Répartition des sentiments - Pie Chart")

    # colonnes et titres
col = 'roberta_label'
title = ''

value_counts = df[col].value_counts()
labels = value_counts.index.tolist()
values = value_counts.values.tolist()

donut_colors = [color_map[label] for label in labels]

    # Création du donut
fig_pie = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,  # Donut style
        marker=dict(colors=donut_colors),
        textinfo='percent',
    )])

    # Layout
fig_pie.update_layout(
        title=title,
        margin=dict(t=50, b=0, l=0, r=0),
        showlegend=False
    )

    # Affichage Streamlit
#st.plotly_chart(fig, use_container_width=True)

col1, col2 = st.columns([2,1])

with col1:
    st.plotly_chart(fig_roberta, use_container_width=True)

with col2:
    st.plotly_chart(fig_pie, use_container_width=True)




'''st.subheader("Timeline")

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
        title='Average Daily Sentiment (VADER)',
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

st.plotly_chart(fig, use_container_width=True)'''



st.subheader("Timeline")

# Vérification que les dates sont bien converties
df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
df['date'] = df['created_at'].dt.date

# Mapper les labels en scores numériques
sentiment_map = {'negative': -1, 'neutral': 0, 'positive': 1}
df['vader_score'] = df['vader_label'].map(sentiment_map)

# Moyenne quotidienne des scores
daily_sentiment = df.groupby('date')['vader_score'].mean().reset_index()

# Affichage
fig = px.line(
    daily_sentiment,
    x='date',
    y='vader_score',
    title='Évolution quotidienne du sentiment moyen (VADER)',
    markers=True,
    labels={'vader_score': 'Sentiment moyen', 'date': 'Date'},
    line_shape='spline',  # Ajoute une courbe lissée
)

fig.update_traces(
    marker=dict(size=6, color='royalblue', line=dict(width=1, color='DarkSlateGrey')),
    line=dict(color='royalblue', width=3)
)

fig.update_layout(
    title_x=0.5,  # Centre le titre
    title_font=dict(size=22),
    xaxis_title='Date',
    yaxis_title='Sentiment moyen',
    xaxis_tickangle=-45,
    yaxis=dict(dtick=0.5, gridcolor='lightgrey'),
    xaxis=dict(showgrid=True, gridcolor='lightgrey'),
    template='plotly_white',
    plot_bgcolor='rgba(0,0,0,0)',  # Fond transparent
)

st.plotly_chart(fig, use_container_width=True)




    # calcul du score moyen par topic
topic_sentiment = df.groupby('topic')['vader_score'].mean().sort_values().reset_index()

    # affichage
fig = px.bar(
        topic_sentiment,
        x='vader_score',
        y='topic',
        orientation='h',
        title='Average Topic Sentiment (RoBERTa)',
        labels={'vader_score': 'Average Sentiment', 'topic': 'Topic'},
        color='vader_score',
        color_continuous_scale='RdYlGn',
    )

fig.update_layout(
        xaxis_title='Average Sentiment',
        yaxis_title='Topic',
        template='plotly_white'
    )

st.plotly_chart(fig, use_container_width=True)

'''# affichage
    figs = []
    for col, title in zip(columns, titles):
        value_counts = df[col].value_counts()
        labels = value_counts.index.tolist()
        values = value_counts.values.tolist()

    fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,  # Donut style
            marker=dict(colors=px.colors.qualitative.Set3),
            textinfo='percent',
    )])

    fig.update_layout(
        title=title,
        margin=dict(t=50, b=0, l=0, r=0),
        showlegend=True
    )

    figs.append(fig)


    cols = st.columns(3)
    for col, fig in zip(cols, figs):
        with col:
            st.plotly_chart(fig, use_container_width=True)'''
    













'''# TextBlob
textblob_counts = df['textblob_label'].value_counts().reset_index()
textblob_counts.columns = ['Sentiment', 'Tweets']
textblob_counts['Color'] = textblob_counts['Sentiment'].map(color_map)

fig_textblob = px.bar(textblob_counts, 
                      x='Sentiment', y='Tweets',
                      title="TextBlob Sentiment Distribution",
                      color='Sentiment', color_discrete_map = color_map)
fig_textblob.update_layout(barmode='stack')

# VADER
vader_counts = df['vader_label'].value_counts().reset_index()
vader_counts.columns = ['Sentiment', 'Tweets']
vader_counts['Color'] = vader_counts['Sentiment'].map(color_map)

fig_vader = px.bar(vader_counts, 
                   x='Sentiment', y='Tweets',
                   title="VADER Sentiment Distribution",
                   color='Sentiment', color_discrete_map = color_map)
fig_vader.update_layout(barmode='stack')'''

#st.plotly_chart(fig_textblob)
#st.plotly_chart(fig_vader)




















