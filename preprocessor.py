# preprocessor.py
import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import os

# Download NLTK resources at module import if not available
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data_whatsapp")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

for pkg in ('punkt', 'wordnet', 'stopwords'):
    try:
        nltk.data.find(f'tokenizers/{pkg}' if pkg == 'punkt' else f'corpora/{pkg}')
    except LookupError:
        nltk.download(pkg, download_dir=nltk_data_dir)

def preprocess(raw_text: str) -> pd.DataFrame:
    """
    Parse WhatsApp exported chat text into a DataFrame with computed fields.
    Assumes WhatsApp format like: '12/12/20, 12:34 - Name: message'
    """
    # Pattern handles dd/mm/yyyy or dd/mm/yy; adjust if WhatsApp locale differs
    pattern = r'(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2})\s-\s'
    # Separate messages by date-time marker
    parts = re.split(pattern, raw_text)
    # parts structure: ['', date, time, rest, date, time, rest, ...] due to split groups
    messages = []
    if len(parts) < 4:
        # fallback: try splitting by newline and removing empty lines
        lines = [l for l in raw_text.splitlines() if l.strip()]
        # minimal DataFrame
        df = pd.DataFrame({'date': pd.to_datetime([]), 'user': [], 'message': []})
        return df

    # Iterate and build message entries
    entries = []
    i = 1
    while i < len(parts):
        date_str = parts[i]
        time_str = parts[i+1]
        rest = parts[i+2]
        dt_str = f"{date_str}, {time_str}"
        entries.append((dt_str, rest.strip()))
        i += 3

    df = pd.DataFrame(entries, columns=['date_raw', 'user_message'])
    # parse datetime with flexible formats
    def parse_dt(s):
        for fmt in ("%d/%m/%Y, %H:%M", "%d/%m/%y, %H:%M"):
            try:
                return pd.to_datetime(s, format=fmt)
            except Exception:
                continue
        return pd.to_datetime(s, errors='coerce')
    df['date'] = df['date_raw'].apply(parse_dt)

    # split user and message; if no user, mark group_notification
    users = []
    msgs = []
    for um in df['user_message']:
        # attempt split "Name: message"
        split = re.split(r'([^:]+):\s', um, maxsplit=1)
        # re.split returns ['','Name',' message'] when pattern matches at start; we'll handle generically
        if len(split) >= 3 and split[1].strip():
            # split[1] is user, split[2] is message
            users.append(split[1].strip())
            msgs.append(split[2].strip())
        else:
            users.append('group_notification')
            msgs.append(um.strip())

    df['user'] = users
    df['message'] = msgs
    df.drop(columns=['date_raw', 'user_message'], inplace=True)

    # add time-derived columns
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    # period strings
    period = []
    for hour in df['hour']:
        if pd.isna(hour):
            period.append(None)
        elif hour == 23:
            period.append(f"{hour}-00")
        elif hour == 0:
            period.append("00-1")
        else:
            period.append(f"{hour}-{hour+1}")
    df['period'] = period

    # Basic cleaning + tokenization + lemmatization
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def clean_text(text):
        if not isinstance(text, str):
            return ""
        # remove urls, non-word characters, extra spaces
        text = re.sub(r'http\S+', ' ', text)
        text = re.sub(r'www.\S+', ' ', text)
        text = re.sub(r'[^A-Za-z0-9\s]+', ' ', text)
        words = word_tokenize(text)
        words = [w for w in words if w.lower() not in stop_words]
        words = [lemmatizer.lemmatize(w) for w in words]
        return " ".join(words)

    df['cleaned_messages'] = df['message'].apply(clean_text)
    return df
