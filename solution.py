import pandas as pd
import re
import emoji
from collections import Counter
from textblob import TextBlob
from urlextract import URLExtract
import spacy
import subprocess

def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")
# nlp = spacy.load('en_core_web_sm')
nlp = load_spacy_model()

def fetch_stats(selected_users, df):
    if selected_users != "Overall":
        df = df[df['users_name'] == selected_users]

    num_messages = df.shape[0]
    words = []
    for message in df['messages']:
        words.extend(message.split())
    num_media = df[df['messages'] == '<Media omitted>\n'].shape[0]
    #fetch number of the links....
    extract = URLExtract()
    links = []
    for message in df['messages']:
        links.extend(extract.find_urls(message))
    return num_messages, len(words), num_media, len(links)


def get_sentiment(selected_users, df):
    if selected_users == 'Overall':
        messages = df['messages'].dropna().astype(str)  # Convert to string
    else:
        messages = df[df['users_name'] == selected_users]['messages'].dropna().astype(str)

    sentiments = messages.apply(lambda msg: analyze_sentiment(msg))
    return sentiments
def analyze_sentiment(message):
    analysis = TextBlob(message)
    polarity = analysis.sentiment.polarity  # Get sentiment score

    # Categorize into Positive, Neutral, Negative
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"


def extract_emojis(text):
    """Extracts all emojis from a given text."""
    return [char for char in text if char in emoji.EMOJI_DATA]
def count_emojis(df, selected_user):
    if selected_user != "Overall":
        df = df[df['users_name'] == selected_user]

    all_messages = ''.join(df['messages'].astype(str))
    all_emojis = []
    for message in all_messages:
        all_emojis.extend(extract_emojis(message))  # Collect all emojis

    emoji_counts = Counter(all_emojis)  # Count occurrences
    return emoji_counts


def get_most_used_word(selected_user, df):
    if selected_user != "Overall":
        df = df[df['users_name'] == selected_user]


    # Convert the column to a single string
    messages = df['messages'].dropna().str.cat(sep=' ')  # Join all messages into one string

    # Process the text
    long_text = nlp(messages)

    # Extract only valid words (no punctuation, no stopwords, no special characters)
    list_of_tokens = [
        token.text.lower() for token in long_text
        if not token.is_stop and not token.is_punct and token.is_alpha  # Ensures only alphabetic words
    ]

    # Count token frequency
    token_frequency = Counter(list_of_tokens)

    return token_frequency


def get_wordcloud(selected_user, df):
    if selected_user != "Overall":
        df = df[df['users_name'] == selected_user]

    text = ' '.join(df['messages'])
    return text
