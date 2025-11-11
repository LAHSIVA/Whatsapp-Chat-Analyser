import re
import pandas as pd
import numpy as np
import emoji
from collections import Counter
from urlextract import URLExtract
from wordcloud import WordCloud
import matplotlib.font_manager as fm
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ---------- NLTK Setup ----------
nltk_data_dir = "nltk_data"  # same as preprocessor
nltk.data.path.append(nltk_data_dir)
nltk.download('vader_lexicon', download_dir=nltk_data_dir, quiet=True)
vader = SentimentIntensityAnalyzer()
extract = URLExtract()

# ---------- Stats Functions ----------
def fetch_stats(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    num_messages = df.shape[0]
    words = sum(len(str(m).split()) for m in df['message'])
    num_media = df[df['message'].str.contains(r'<Media omitted>', na=False)].shape[0]
    links = [url for msg in df['message'].astype(str) for url in extract.find_urls(msg)]
    return num_messages, words, num_media, len(links)

def most_busy_users(df):
    counts = df['user'].value_counts().head(10)
    percent_df = round((df['user'].value_counts() / df.shape[0]) * 100, 2).reset_index()
    percent_df.rename(columns={'index': 'name', 'user': 'percent'}, inplace=True)
    return counts, percent_df

# ---------- WordCloud ----------
def create_wordcloud(selected_user, df, stop_words_file='stop_hinglish.txt'):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    temp = df[df['user'] != 'group_notification']
    temp = temp[~temp['message'].str.contains(r'<Media omitted>', na=False)]
    
    stop_words = set()
    try:
        with open(stop_words_file, 'r', encoding='utf-8') as f:
            stop_words = set(w.strip() for w in f if w.strip())
    except Exception:
        pass

    temp['message'] = temp['message'].apply(lambda msg: " ".join([w for w in str(msg).lower().split() if w not in stop_words]))
    text = temp['message'].str.cat(sep=" ")
    try:
        wc = WordCloud(width=500, height=500, background_color='white', collocations=False)
        return wc.generate(text)
    except Exception:
        default_font = fm.findfont(fm.FontProperties(family='DejaVu Sans'))
        wc = WordCloud(width=500, height=500, background_color='white', font_path=default_font, collocations=False)
        return wc.generate(text)

# ---------- Common Words ----------
def most_common_words(selected_user, df, top_n=20, stop_words_file='stop_hinglish.txt'):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    temp = df[df['user'] != 'group_notification']
    
    stop_words = set()
    try:
        with open(stop_words_file, 'r', encoding='utf-8') as f:
            stop_words = set(w.strip() for w in f if w.strip())
    except Exception:
        pass

    words = [w for msg in temp['message'].astype(str) for w in msg.lower().split() if w not in stop_words]
    return pd.DataFrame(Counter(words).most_common(top_n), columns=['word', 'count'])

# ---------- Emojis ----------
def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    emojis = [e for msg in df['message'].astype(str) for e in re.findall(emoji.get_emoji_regexp(), msg)]
    return pd.DataFrame(Counter(emojis).most_common(), columns=['emoji', 'count'])

# ---------- Timelines ----------
def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    timeline = df.groupby(['year','month_num','month']).count()['message'].reset_index()
    timeline['time'] = timeline.apply(lambda r: f"{r['month']}-{r['year']}", axis=1)
    return timeline[['time','message']]

def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    daily = df.groupby('only_date').count()['message'].reset_index()
    daily.rename(columns={'message':'message'}, inplace=True)
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
        heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)
        order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        heatmap = heatmap.reindex(order).fillna(0)
        return heatmap
    except Exception:
        return None

# ---------- Sentiment ----------
def analyze_emotion(user_input: str):
    scores = vader.polarity_scores(user_input)
    if scores['compound'] >= 0.05:
        return "Positive"
    elif scores['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def analyze_emotions_for_users(df):
    emotions = {}
    for user in df['user'].unique():
        if user == 'group_notification':
            continue
        emotions[user] = [analyze_emotion(str(m)) for m in df[df['user']==user]['message']]
    return emotions
