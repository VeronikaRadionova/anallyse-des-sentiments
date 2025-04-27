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



# Importation du data dans le repertoire CSV
dataframes = {'tweets_with_sentiments'}
labels = {
    "TRECIS-CTIT-H-001": "fireColorado2012",
    "TRECIS-CTIT-H-002": "costaRicaEarthquake2012",
    "TRECIS-CTIT-H-003": "floodColorado2013",
    "TRECIS-CTIT-H-004": "typhoonPablo2012",
    "TRECIS-CTIT-H-005": "laAirportShooting2013",
    "TRECIS-CTIT-H-006": "westTexasExplosion2013",
    "TRECIS-CTIT-H-007": "guatemalaEarthquake2012",
    "TRECIS-CTIT-H-008": "italyEarthquakes2012",
    "TRECIS-CTIT-H-009": "philipinnesFloods2012",
    "TRECIS-CTIT-H-010": "albertaFloods2013",
    "TRECIS-CTIT-H-011": "australiaBushfire2013",
    "TRECIS-CTIT-H-012": "bostonBombings2013",
    "TRECIS-CTIT-H-013": "manilaFloods2013",
    "TRECIS-CTIT-H-014": "queenslandFloods2013",
    "TRECIS-CTIT-H-015": "typhoonYolanda2013",
    "TRECIS-CTIT-H-016": "joplinTornado2011",
    "TRECIS-CTIT-H-017": "chileEarthquake2014",
    "TRECIS-CTIT-H-018": "typhoonHagupit2014",
    "TRECIS-CTIT-H-019": "nepalEarthquake2015",
    "TRECIS-CTIT-H-020": "flSchoolShooting2018",
    "TRECIS-CTIT-H-021": "parisAttacks2015",
    "TRECIS-CTIT-H-022": "floodChoco2019",
    "TRECIS-CTIT-H-023": "fireAndover2019",
    "TRECIS-CTIT-H-024": "earthquakeCalifornia2014",
    "TRECIS-CTIT-H-025": "earthquakeBohol2013",
    "TRECIS-CTIT-H-026": "hurricaneFlorence2018",
    "TRECIS-CTIT-H-027": "shootingDallas2017",
    "TRECIS-CTIT-H-028": "fireYMM2016",
    "TRECIS-CTIT-H-029": "albertaWildfires2019",
    "TRECIS-CTIT-H-030": "cycloneKenneth2019",
    "TRECIS-CTIT-H-031": "philippinesEarthquake2019",
    "TRECIS-CTIT-H-032": "coloradoStemShooting2019",
    "TRECIS-CTIT-H-033": "southAfricaFloods2019",
    "TRECIS-CTIT-H-034": "sandiegoSynagogueShooting2019"
}

label_to_code = {v: k for k, v in labels.items()}




# MAIN AFFICHAGE

def analyse_sentiments(dataframes, labels): 
    st.title("Analyse des Sentiments des Tweets")

    if "tweets_with_sentiments" not in dataframes:
        st.error("tweets_with_sentiments est manquant dans les données chargées")
        return

    df = pd.read_csv('tweets_with_sentiments.csv')

    # # #
    crises = df["topic"].dropna().unique()
    crises_lisibles = [labels.get(code, code) for code in sorted(crises)]
    selected_label = st.selectbox("Choisissez une crise 📍", crises_lisibles)
    selected_crisis = label_to_code.get(selected_label, selected_label)

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

    col1, col2 = st.columns([2,1])

    with col1:
        st.plotly_chart(fig_roberta, use_container_width=True)

    with col2:
        st.plotly_chart(fig_pie, use_container_width=True)




    st.subheader("Timeline")

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
    



analyse_sentiments(dataframes,labels)









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




















