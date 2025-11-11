import re
import os
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# ---------- NLTK Setup ----------
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download necessary packages if missing
for pkg in ["punkt", "wordnet", "stopwords", "vader_lexicon"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}" if pkg=="punkt" else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, download_dir=nltk_data_dir, quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess(raw_text: str) -> pd.DataFrame:
    pattern = r'(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2})\s-\s'
    parts = re.split(pattern, raw_text)

    if len(parts) < 4:
        return pd.DataFrame({'date': pd.to_datetime([]), 'user': [], 'message': []})

    entries = []
    i = 1
    while i < len(parts):
        date_str, time_str, rest = parts[i], parts[i+1], parts[i+2]
        dt_str = f"{date_str}, {time_str}"
        entries.append((dt_str, rest.strip()))
        i += 3

    df = pd.DataFrame(entries, columns=['date_raw', 'user_message'])

    def parse_dt(s):
        for fmt in ("%d/%m/%Y, %H:%M", "%d/%m/%y, %H:%M"):
            try:
                return pd.to_datetime(s, format=fmt)
            except Exception:
                continue
        return pd.to_datetime(s, errors='coerce')

    df['date'] = df['date_raw'].apply(parse_dt)

    users, msgs = [], []
    for um in df['user_message']:
        split = re.split(r'([^:]+):\s', um, maxsplit=1)
        if len(split) >= 3 and split[1].strip():
            users.append(split[1].strip())
            msgs.append(split[2].strip())
        else:
            users.append('group_notification')
            msgs.append(um.strip())

    df['user'] = users
    df['message'] = msgs
    df.drop(columns=['date_raw', 'user_message'], inplace=True)

    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['period'] = df['hour'].apply(lambda h: f"{h}-{(h+1)%24}" if pd.notna(h) else None)

    # Cleaning + tokenization + lemmatization
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r'http\S+', ' ', text)
        text = re.sub(r'www.\S+', ' ', text)
        text = re.sub(r'[^A-Za-z0-9\s]+', ' ', text)
        words = word_tokenize(text)  # uses default punkt, no punkt_tab
        words = [lemmatizer.lemmatize(w) for w in words if w.lower() not in stop_words]
        return " ".join(words)

    df['cleaned_messages'] = df['message'].apply(clean_text)
    return df
