import os
import requests
import streamlit as st
from dotenv import load_dotenv
import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents.run import RunConfig
from datetime import datetime


GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
NEWS_API_KEY = st.secrets["NEWS_API_KEY"]

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY is not set in .env file.")
    st.stop()

if not NEWS_API_KEY:
    st.error("NEWS_API_KEY is not set in .env file.")
    st.stop()

# Gemini setup
external_client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-1.5-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

# Fetch conflict news using NewsAPI
def fetch_conflict_news(query):
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&pageSize=5&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    articles = response.json().get("articles", [])

    if not articles:
        return "🚫 No recent news found for this topic."

    summary = f"🔍 **{query.upper()} Conflict Report**\n\n"
    for article in articles:
        title = article['title']
        source = article['source']['name']
        url = article['url']
        published = article['publishedAt'][:10]
        summary += f"- **{title}**\n  ({source}, {published})\n  [Read more]({url})\n\n"

    return summary

# Run Gemini AI Agent
async def run_conflict_agent(prompt):
    agent = Agent(
        name="ConflictTracker",
        instructions="You are ConflictTracker AI, an expert in summarizing and explaining global conflict and unrest data. Respond in clear and accurate detail with facts and figures.",
        model=model
    )
    result = await Runner.run(agent, prompt, run_config=config)
    return result.final_output

# Streamlit UI
st.set_page_config(page_title="ConflictTracker AI Agent", page_icon="🛡️")
st.title("🛡️ ConflictTracker – Global Conflict Monitoring AI")
st.write("This AI agent helps monitor and explain recent political unrest, war, and conflict situations across the world using real-time news and AI analysis.")

# Show Real-time Conflict News
st.subheader("📡 Get Latest Conflict News")
query_input = st.text_input("Enter a country, region, or conflict topic (e.g., Iran Israel, Sudan, Kashmir):")
if st.button("🔎 Fetch Conflict News") and query_input:
    with st.spinner("Fetching news and generating insights..."):
        news = fetch_conflict_news(query_input)
        st.markdown(news)

        # AI summary based on the same topic
        ai_prompt = f"""
You are a geopolitical analyst.
Provide a detailed, factual, and updated report on the current conflict or situation related to: {query_input}.
Include background, causes, key players, recent events, casualties (if any), and international reactions.
"""
        response = asyncio.run(run_conflict_agent(ai_prompt))
        st.success(response)

# User Chat Section
st.markdown("---")
st.subheader("💬 Ask Anything About Conflicts")
user_query = st.text_input("Ask ConflictTracker AI anything (e.g., Will there be war between X and Y?):")
if user_query:
    with st.spinner("Analyzing with AI..."):
        response = asyncio.run(run_conflict_agent(user_query))
        st.success(response)
