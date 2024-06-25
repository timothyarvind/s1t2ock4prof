import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.express as px
import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import feedparser

# Download the VADER lexicon
nltk.download('vader_lexicon')

# Function to fetch news articles from Yahoo Finance RSS feed
def fetch_news(query):
    url = f"https://finance.yahoo.com/rss/headline?s={query}"
    feed = feedparser.parse(url)
    news_data = []
    for entry in feed.entries:
        headline = entry.title
        link = entry.link
        snippet = entry.summary
        news_data.append({'Headline': headline, 'Link': link, 'Snippet': snippet})
    return pd.DataFrame(news_data)

# Function to perform sentiment analysis using VADER
def analyze_sentiment_vader(text):
    sid = SentimentIntensityAnalyzer()
    sentiment = sid.polarity_scores(text)
    if sentiment['compound'] >= 0.5:
        return "Extremely Positive"
    elif sentiment['compound'] > 0:
        return "Positive"
    elif sentiment['compound'] <= -0.5:
        return "Extremely Negative"
    elif sentiment['compound'] < 0:
        return "Negative"
    else:
        return "Neutral"

# Function to get financial data
@st.cache_data
def get_financials(ticker):
    try:
        stock = yf.Ticker(ticker)
        financials = stock.financials
        return financials
    except Exception as e:
        st.error(f"Failed to retrieve data for ticker {ticker}: {e}")
        return None

# Function to calculate financial metrics for a list of tickers
def calculate_financial_metrics(tickers, metrics):
    financial_data = {ticker: pd.DataFrame() for ticker in tickers}
    
    for ticker in tickers:
        financials = get_financials(ticker)
        if financials is None:
            continue
        data = []
        for year in financials.columns:
            year_dt = pd.to_datetime(year).year
            row = {'Year': year_dt}
            for item in financials.index:
                row[item] = financials.loc[item][year]
            data.append(row)
        financial_data[ticker] = pd.DataFrame(data)
    
    metric_data = pd.DataFrame(columns=['Year', 'Metric', 'Ticker', 'Value'])
    
    for ticker in tickers:
        df = financial_data[ticker]
        if df.empty:
            continue
        for metric in metrics:
            for _, row in df.iterrows():
                year = row['Year']
                total_revenue = row.get('Total Revenue', None)
                net_income = row.get('Net Income', None)
                gross_profit = row.get('Gross Profit', None)
                operating_income = row.get('Operating Income', None)
                if pd.notnull(total_revenue):
                    if metric == 'Gross Profit Margin' and pd.notnull(gross_profit):
                        value = (gross_profit / total_revenue) * 100
                    elif metric == 'Net Profit Margin' and pd.notnull(net_income):
                        value = (net_income / total_revenue) * 100
                    elif metric == 'Operating Margin' and pd.notnull(operating_income):
                        value = (operating_income / total_revenue) * 100
                    else:
                        continue
                    metric_data = pd.concat([metric_data, pd.DataFrame({'Year': [int(year)], 'Metric': [metric], 'Ticker': [ticker], 'Value': [value]})], ignore_index=True)
    
    metric_data['Year'] = metric_data['Year'].astype(str)
    return metric_data

# Streamlit app
st.set_page_config(page_title="Stock Data Dashboard", page_icon="📊", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Financial Metrics Comparison", "Sentiment Analysis"])

if page == "Financial Metrics Comparison":
    st.title('Financial Metrics Comparison')

    tickers_input = st.text_input('Enter comma-separated ticker symbols (e.g., AAPL, MSFT, GOOGL):')
    metrics = st.multiselect('Select financial metrics:', ['Gross Profit Margin', 'Net Profit Margin', 'Operating Margin'])

    custom_colors = px.colors.qualitative.Vivid

    if tickers_input and metrics:
        tickers = [ticker.strip().upper() for ticker in tickers_input.split(',')]
        if tickers:
            with st.spinner(f'Calculating selected metrics...'):
                metric_data = calculate_financial_metrics(tickers, metrics)
                if not metric_data.empty:
                    st.subheader('Financial Metrics Table')
                    st.dataframe(metric_data)

                    fig = px.bar(metric_data, x='Year', y='Value', color='Ticker', 
                                 facet_row='Metric', barmode='group', title='Financial Metrics by Year',
                                 height=1000, facet_row_spacing=0.15, color_discrete_sequence=custom_colors)
                    
                    for i, metric in enumerate(metrics):
                        fig.update_yaxes(title_text=metric, row=i+1, col=1)

                    fig.update_traces(texttemplate='%{y:.2f}%', textposition='outside')

                    fig.update_layout(margin=dict(t=100))

                    st.subheader('Financial Metrics Chart')
                    st.plotly_chart(fig)
                else:
                    st.error("No data available for the given ticker symbols and metrics.")

    st.markdown("This app calculates and visualizes selected financial metrics for a set of companies over the years. Enter the ticker symbols of the companies you're interested in, select the financial metrics, and see the results.")

elif page == "Sentiment Analysis":
    st.title('Stock Sentiment Analysis')

    query = st.text_input('Enter a stock ticker symbol to analyze sentiment (e.g., AAPL):')

    if query:
        with st.spinner(f'Fetching news for {query}...'):
            news_df = fetch_news(query)
            st.write(f"Fetched {len(news_df)} news articles")

            if not news_df.empty:
                with st.spinner('Analyzing sentiment...'):
                    news_df['Sentiment'] = news_df['Snippet'].apply(analyze_sentiment_vader)
                    
                    st.subheader('News Articles')
                    st.dataframe(news_df)

                    fig = px.histogram(news_df, x='Sentiment', color='Sentiment', 
                                       title=f'Sentiment Analysis for {query} News Articles')
                    st.plotly_chart(fig)
            else:
                st.error("No news articles found for the given query.")
    else:
        st.info("Enter a stock ticker symbol to fetch and analyze news articles.")
