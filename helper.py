# helper.py
import pandas as pd
from urlextract import URLExtract
from collections import Counter
from wordcloud import WordCloud
import matplotlib.font_manager as fm
import re
import nltk
import emoji
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize.punkt import PunktSentenceTokenizer
tokenizer = PunktSentenceTokenizer()  # default English

nltk.download('punkt', quiet=True)
nltk.download('vader_lexicon', quiet=True)
extract = URLExtract()
vader = SentimentIntensityAnalyzer()

def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    num_messages = df.shape[0]
    words = sum(len(str(m).split()) for m in df['message'])
    num_media_messages = df[df['message'].str.contains(r'<Media omitted>|<Media omitted>\n', na=False)].shape[0]
    links = []
    for message in df['message'].astype(str):
        links.extend(extract.find_urls(message))
    return num_messages, words, num_media_messages, len(links)

def most_busy_users(df):
    x = df['user'].value_counts().head(10)
    percent_df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index().rename(columns={'index': 'name', 'user': 'percent'})
    return x, percent_df

def create_wordcloud(selected_user, df, stop_words_file='stop_hinglish.txt'):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    temp = df[df['user'] != 'group_notification']
    temp = temp[~temp['message'].str.contains(r'<Media omitted>|<Media omitted>\n', na=False)]
    # load stop words
    stop_words = set()
    try:
        with open(stop_words_file, 'r', encoding='utf-8') as f:
            stop_words = set([w.strip() for w in f.read().splitlines() if w.strip()])
    except Exception:
        stop_words = set()

    def remove_stop_words(message):
        if not isinstance(message, str):
            return ""
        return " ".join([w for w in message.lower().split() if w not in stop_words])

    temp = temp.copy()
    temp['message'] = temp['message'].apply(remove_stop_words)
    text = temp['message'].str.cat(sep=" ")
    try:
        wc = WordCloud(width=500, height=500, background_color='white', collocations=False)
        return wc.generate(text)
    except Exception:
        default_font = fm.findfont(fm.FontProperties(family='DejaVu Sans'))
        wc = WordCloud(width=500, height=500, background_color='white', font_path=default_font, collocations=False)
        return wc.generate(text)

def most_common_words(selected_user, df, top_n=20, stop_words_file='stop_hinglish.txt'):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    temp = df[df['user'] != 'group_notification']
    stop_words = set()
    try:
        with open(stop_words_file, 'r', encoding='utf-8') as f:
            stop_words = set([w.strip() for w in f.read().splitlines() if w.strip()])
    except Exception:
        stop_words = set()

    words = []
    for message in temp['message'].astype(str):
        for w in message.lower().split():
            if w not in stop_words:
                words.append(w)
    return pd.DataFrame(Counter(words).most_common(top_n), columns=['word', 'count'])

def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    emojis = []
    for message in df['message'].astype(str):
        emojis.extend(re.findall(emoji.get_emoji_regexp(), message))
    return pd.DataFrame(Counter(emojis).most_common(), columns=['emoji', 'count'])

def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()
    timeline['time'] = timeline.apply(lambda row: f"{row['month']}-{row['year']}", axis=1)
    return timeline[['time', 'message']]

def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    daily = df.groupby('only_date').count()['message'].reset_index()
    daily = daily.rename(columns={'only_date': 'only_date', 'message': 'message'})
    return daily

def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['day_name'].value_counts()

def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df['month'].value_counts()

def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    try:
        user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
        # Reorder day names standard order
        order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        user_heatmap = user_heatmap.reindex(order).fillna(0)
        return user_heatmap
    except Exception:
        return None

# Simple VADER-based emotion classification fallback
def analyze_emotion(user_input: str):
    scores = vader.polarity_scores(user_input)
    if scores['compound'] >= 0.05:
        return "Positive"
    elif scores['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def analyze_emotions_for_users(df):
    user_emotions = {}
    for user in df['user'].unique():
        if user == 'group_notification':
            continue
        user_df = df[df['user'] == user]
        emotions = [analyze_emotion(str(m)) for m in user_df['message']]
        user_emotions[user] = emotions
    return user_emotions