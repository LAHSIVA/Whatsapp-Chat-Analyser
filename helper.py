from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import re
import emoji
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()
extract=URLExtract()
def fetch_stats(selected_user,df):
    f = open('stop_hinglish.txt','r')
    stop_words = f.read()
    if selected_user!='Overall':
        df=df[df['user'] == selected_user]

    num_messages=df.shape[0]
    words=[]
    for message in df['message']:
        words.extend(message.split())

    #fetch number of media messages
    num_media_messages=df[df['message']=='<Media omitted>\n'].shape[0]

    #fetch number of links shared

    links= []
    for message in df['message']:
        links.extend(extract.find_urls(message))

    return num_messages,len(words),num_media_messages,len(links)

def most_busy_users(df):
    x= df['user'].value_counts().head()
    df=round((df['user'].value_counts()/df.shape[0])*100,2).reset_index().rename(
        columns={'index':'name','user':'percent'})
    return x,df

def create_wordcloud(selected_user,df):

    f = open('stop_hinglish.txt', 'r')
    stop_words = f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[df['user'] != 'group_notification']
    temp = temp[temp['message'] != '<Media omitted>\n']

    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)

    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)

    df_wc=wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

def most_common_words(selected_user,df):
    f = open('stop_hinglish.txt','r')
    stop_words = f.read()
    if selected_user !='Overall':
        df=df[df['user']== selected_user]

    temp=df[df['user'] != 'group_notification']
    temp=temp[temp['message'] !='<Media omitted>\n']

    words=[]

    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)
    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def emoji_helper(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emoji_list = re.findall(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U0001FB00-\U0001FBFF\U0001FC00-\U0001FCFF\U0001FD00-\U0001FDFF\U0001FE00-\U0001FEFF\U0001FF00-\U0001FFFF]', message)
        emojis.extend(emoji_list)

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))

    return emoji_df

def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    timeline = df.groupby(['year', 'month_num', 'month']).count()['message'].reset_index()

    # Combine 'month' and 'year' columns to create the 'time' column
    timeline['time'] = timeline.apply(lambda row: f"{row['month']}-{row['year']}", axis=1)

    return timeline


def daily_timeline(selected_user,df):
    if selected_user !='Overall':
        df = df[df['user'] == selected_user]

    daily_timeline = df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def activity_heatmap(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    user_heatmap = df.pivot_table(index='day_name', columns='period', values='message', aggfunc='count').fillna(0)

    return user_heatmap
# Define a function to analyze emotions
def analyze_emotion(user_input):
    sentiment_scores = analyzer.polarity_scores(user_input)

    if sentiment_scores['compound'] >= 0.05:
        return "Positive"
    elif sentiment_scores['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

# Define a function to analyze emotions for all users
def analyze_emotions_for_users(df):
    user_emotions = {}

    for user in df["user"].unique():
        if user != "group_notification":
            user_df = df[df["user"] == user]
            emotions = []

            for message in user_df["message"]:
                emotion = analyze_emotion(message)
                emotions.append(emotion)

            user_emotions[user] = emotions

    return user_emotions