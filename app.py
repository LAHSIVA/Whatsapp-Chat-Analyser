import streamlit as st
import preprocessor
import helper
import gensim
import matplotlib.pyplot as plt
import seaborn as sns
from gensim import corpora, models
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from emotion_analyzer import analyze_emotion

# Streamlit app
st.title("WhatsApp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    if st.sidebar.button("Show Analysis"):
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)

        col1, col2, col3, col4 = st.columns(4)
        st.title("Top Statistics")

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links shared")
            st.title(num_links)

        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation=90)  # Correct xticks rotation
        st.pyplot(fig)

        st.title('Daily Timeline')
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation=90)  # Correct xticks rotation
        st.pyplot(fig)

        st.title('Activity Map')
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most busy day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values)
            plt.xticks(rotation=90)  # Correct xticks rotation
            st.pyplot(fig)

        with col2:
            st.header("Most busy month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation=90)  # Correct xticks rotation
            st.pyplot(fig)

        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        sns.heatmap(user_heatmap, ax=ax)  # Use sns.heatmap instead of ax = sns.heatmap
        st.pyplot(fig)

        if selected_user == 'Overall':
            st.title('Most Busy Users')
            x, new_df = helper.most_busy_users(df)
            fig, ax = plt.subplots()
            ax.bar(x.index, x.values)
            plt.xticks(rotation=90)  # Correct xticks rotation
            st.pyplot(fig)
            col1, col2 = st.columns(2)

            with col1:
                ax.bar(x.index, x.values, color='red')
                plt.xticks(rotation=90)  # Correct xticks rotation
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc, interpolation="bilinear")
        st.pyplot(fig)

        most_common_df = helper.most_common_words(selected_user, df)
        fig, ax = plt.subplots()
        ax.bar(most_common_df[0], most_common_df[1])
        plt.xticks(rotation=90)  # Correct xticks rotation
        st.title('Most Common Words')
        st.pyplot(fig)

        if selected_user == 'Overall':
            st.title("Topic Modeling (LDA)")
            lda_num_topics = st.number_input("Number of Topics (LDA)", min_value=1, max_value=10, value=5)
            if st.button("Run LDA"):
                try:
                    st.write("Running LDA for Topic Modeling...")
                    messages = df['cleaned_messages'].values.tolist()
                    dictionary = corpora.Dictionary([message.split() for message in messages])
                    corpus = [dictionary.doc2bow(message.split()) for message in messages]
                    lda_model = gensim.models.LdaModel(corpus, num_topics=lda_num_topics, id2word=dictionary, passes=15)
                    topics = lda_model.print_topics(num_words=5)
                    st.write("LDA Topics:")
                    for topic in topics:
                        st.write(topic)
                except Exception as e:
                    st.write(f"An error occurred during LDA modeling: {str(e)}")

            st.title("Topic Modeling (NMF)")
            nmf_num_topics = st.number_input("Number of Topics (NMF)", min_value=1, max_value=10, value=5)
            if st.button("Run NMF"):
                try:
                    st.write("Running NMF for Topic Modeling...")
                    messages = df['cleaned_messages'].values.tolist()
                    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
                    tfidf = tfidf_vectorizer.fit_transform(messages)
                    nmf = NMF(n_components=nmf_num_topics, random_state=1)
                    nmf.fit(tfidf)
                    feature_names = tfidf_vectorizer.get_feature_names_out()
                    st.write("NMF Topics:")
                    for topic_idx, topic in enumerate(nmf.components_):
                        top_words_idx = topic.argsort()[:-6:-1]
                        top_words = [feature_names[i] for i in top_words_idx]
                        st.write(f"Topic #{topic_idx}: {' '.join(top_words)}")
                except Exception as e:
                    st.write(f"An error occurred during NMF modeling: {str(e)}")

        st.title("Emotion Analysis")

        st.subheader("Personal Emotions")
        user_list = df['user'].unique().tolist()
        user_list.remove('group_notification')
        user_list.sort()
        user_list.insert(0, "Overall")

        selected_user = st.selectbox("Select a user:", user_list)

        if selected_user != "Overall":
            user_df = df[df['user'] == selected_user]
            st.subheader(f"Emotions for {selected_user}:")

            # Input text area for emotion analysis
            user_input = st.text_area(f"Enter a text message for {selected_user}:")
            if st.button(f"Analyze Emotion for {selected_user}"):
                result = analyze_emotion(user_input)
                st.write(result)

            def analyze_user_emotion():
                if user_input:
                    predicted_emotion = analyze_emotion(user_input)
                    st.write(f"Predicted Emotion for {selected_user}: {predicted_emotion}")
                else:
                    st.warning(f"Please enter a text message for {selected_user} analysis.")

            if st.button(f"Analyze Emotion for {selected_user}"):
                analyze_user_emotion()

        st.subheader("Overall Chat Emotion")
        # Input text area for overall chat emotion analysis
        chat_input = st.text_area("Enter a text message for overall chat:")

        def analyze_chat_emotion():
            if chat_input:
                predicted_emotion = analyze_emotion(chat_input)
                st.write(f"Predicted Emotion for Overall Chat: {predicted_emotion}")
            else:
                st.warning("Please enter a text message for overall chat analysis.")

        if st.button("Analyze Emotion for Overall Chat"):
            analyze_chat_emotion()
