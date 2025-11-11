# app.py
import os
from dotenv import load_dotenv
load_dotenv()  # ✅ Load .env before anything else that depends on env vars

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessor import preprocess
import helper
from rag_module import Retriever, generate_with_groq

# Streamlit page config
st.set_page_config(page_title="WhatsApp Chat Analyzer + RAG (Groq)", layout="wide")

st.title("📊 WhatsApp Chat Analyzer — Analytics + RAG (Groq Cloud)")


uploaded_file = st.sidebar.file_uploader("📁 Upload WhatsApp chat (.txt)", type=["txt"])
if uploaded_file is not None:
    raw = uploaded_file.getvalue().decode('utf-8', errors='ignore')
    df = preprocess(raw)

    if df.empty:
        st.error("❌ Could not parse the uploaded chat file. Ensure it's a standard WhatsApp export.")
        st.stop()

    # user selection
    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0, "Overall")
    selected_user = st.sidebar.selectbox("Show analysis with respect to:", user_list)

    if st.sidebar.button("Show Analysis"):
        # --- BASIC STATS ---
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Messages", num_messages)
        col2.metric("Total Words", words)
        col3.metric("Media Shared", num_media_messages)
        col4.metric("Links Shared", num_links)

        # --- MONTHLY TIMELINE ---
        st.subheader("🗓 Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        if not timeline.empty:
            fig, ax = plt.subplots()
            ax.plot(np.array(timeline['time']), np.array(timeline['message']), marker='o')
            ax.set_xlabel("Month-Year")
            ax.set_ylabel("Messages")
            plt.xticks(rotation=90)
            st.pyplot(fig)
        else:
            st.info("No monthly data available.")

        # --- DAILY TIMELINE ---
        st.subheader("📅 Daily Timeline")
        daily = helper.daily_timeline(selected_user, df)
        if not daily.empty:
            fig, ax = plt.subplots()
            ax.plot(np.array(daily['only_date'].astype(str)), np.array(daily['message']), marker='o')
            ax.set_xlabel("Date")
            ax.set_ylabel("Messages")
            plt.xticks(rotation=90)
            st.pyplot(fig)
        else:
            st.info("No daily data available.")

        # --- ACTIVITY MAP ---
        st.subheader("📈 Activity Map")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Most Busy Day")
            busy_day = helper.week_activity_map(selected_user, df)
            st.bar_chart(busy_day)
        with col2:
            st.write("Most Busy Month")
            busy_month = helper.month_activity_map(selected_user, df)
            st.bar_chart(busy_month)

        # --- HEATMAP ---
        st.subheader("🔥 Weekly Activity Heatmap")
        heatmap = helper.activity_heatmap(selected_user, df)
        if heatmap is not None and not heatmap.empty:
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.heatmap(heatmap, ax=ax, cmap="YlGnBu")
            st.pyplot(fig)
        else:
            st.info("No sufficient data for heatmap.")

        # --- MOST BUSY USERS ---
        if selected_user == 'Overall':
            st.subheader("🏆 Most Busy Users")
            x, newdf = helper.most_busy_users(df)
            fig, ax = plt.subplots()
            ax.bar(x.index.astype(str), x.values, color='teal')
            plt.xticks(rotation=90)
            st.pyplot(fig)
            st.dataframe(newdf)

        # --- WORDCLOUD ---
        st.subheader("☁️ Word Cloud")
        wc = helper.create_wordcloud(selected_user, df)
        if wc:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        else:
            st.info("No valid text for WordCloud.")

        # --- MOST COMMON WORDS ---
        st.subheader("💬 Most Common Words")
        mc = helper.most_common_words(selected_user, df)
        if not mc.empty:
            st.dataframe(mc)
        else:
            st.info("No words found for this selection.")

        # --- EMOTION ANALYSIS ---
        st.subheader("😊 Emotion Analysis (VADER)")
        sample_input = st.text_area("Enter text to classify emotion (optional):")
        if st.button("Classify Emotion"):
            if sample_input:
                emo = helper.analyze_emotion(sample_input)
                st.success(f"Predicted Emotion: **{emo}**")
            else:
                st.write("Emotion distribution (sample):")
                ue = helper.analyze_emotions_for_users(df)
                st.write({
                    k: (sum(1 for e in v if e == 'Positive') / len(v) if v else 0)
                    for k, v in ue.items()
                })

    # ------------------- RAG + GROQ CLOUD -------------------
    st.header("💡 Chat Intelligence (RAG + Groq Cloud)")
    st.markdown("Build a small knowledge base from chat messages (embeddings) and ask contextual questions powered by **Groq Cloud API**.")

    if 'retriever' not in st.session_state:
        st.session_state['retriever'] = None

    if st.button("🧠 Build Knowledge Base (Embeddings)"):
        messages = df['cleaned_messages'].astype(str).tolist() if 'cleaned_messages' in df.columns else df['message'].astype(str).tolist()
        retriever = Retriever()
        with st.spinner("Building embeddings..."):
            retriever.build(messages)
        st.session_state['retriever'] = retriever
        st.success("✅ Knowledge base built successfully!")

    query = st.text_input("Ask a question about the chat (e.g. 'Who talked about project updates?')")
    top_k = st.slider("Number of context messages to retrieve", 1, 10, 5)

    if st.button("🚀 Get Answer") and query:
        retriever = st.session_state.get('retriever')
        if retriever is None:
            st.error("⚠️ Please build the knowledge base first.")
        else:
            retrieved = retriever.query(query, top_k=top_k)
            st.subheader("🔍 Retrieved Context")
            for idx, doc in enumerate(retrieved, 1):
                st.markdown(f"**{idx}.** {doc}")
            try:
                answer = generate_with_groq(query, retrieved)
                st.subheader("🤖 Groq Cloud Answer")
                st.success(answer)
            except Exception as e:
                st.error(f"❌ Error from Groq API: {e}")

else:
    st.info("👆 Upload a WhatsApp exported chat (.txt) to begin analysis.")
