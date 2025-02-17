# AI-Driven Stock Price Forecasting Using NLP and Time Series Models
This project explores whether combining historical stock data with financial news sentiment can improve stock price predictions. Most forecasting models focus only on past trends, but this project looks at how market sentiment captured from news might impact predictions. 

The analysis focuses on 10 stocks: Apple (AAPL), Microsoft (MSFT), NVIDIA (NVDA), Amazon (AMZN), Alphabet (GOOGL), Berkshire Hathaway (BRK.B), Eli Lilly (LLY), Broadcom (AVGO), Tesla (TSLA), and Walmart (WMT)

## Data Collection

The first step of any data science project is to collect the data. We gathered data from multiple sources, including Polygon, GNews, Newsdata, and Nasdaq, for both historical stock data and news data. The data spans from **2021-01-01 to 2024-08-30**, ensuring a robust timeline for analysis.

### Stock Data
Historical stock data was collected using the Polygon API. The data is at a **minute-level granularity**, providing high-resolution insights into market movements. In addition to basic historical data, we also obtained technical indicators to enrich the dataset. These indicators, such as Simple Moving Averages (SMA), Exponential Moving Averages (EMA), Relative Strength Index (RSI), and Moving Average Convergence Divergence (MACD), provide deeper insights into market trends and stock performance. The numerical values in the technical indicators, such as SMA_5 or RSI_7, represent their respective window sizes (e.g., 5-day or 7-day averages). Below are the columns that we fetched using the API:

- `Stock Symbol`
- `Company Name`
- `Timestamp`
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`
- `Volume Weighted Average Price`
- `Number of Transactions`
- Technical Indicators:
  - `SMA_5`, `SMA_10`, `SMA_20`, `SMA_50`, `SMA_60`
  - `EMA_9`, `EMA_12`, `EMA_26`
  - `RSI_7`, `RSI_14`, `RSI_21`, `RSI_30`
  - `MACD_Value`, `MACD_Histogram`, `MACD_Signal`

### News Articles
For news data, we collected over 50,000 financial news articles from various sources such as GNews, Newsdata, and by scraping Nasdaq.com. The data spans the same timeframe as the stock data, from **2021-01-01 to 2024-08-30**, ensuring consistency in the dataset. Each article included the following details:

- `Stock Symbol`
- `Company Name`
- `Timestamp`
- `Category`
- `Title`
- `Description`
- `Image URL`
- `Article URL`
- `Author`
- `Content`

## Sentiment Analysis

Once the data collection was complete, the next step was to analyze the sentiment of the financial news articles. Sentiment analysis helps interpret the tone and emotional context of news data, providing insights into how the market may react to certain events or trends. Here's how we performed sentiment analysis:

- **Tool Used for Extracting Sentiment from the News**:  
  We used FinBERT, a pre-trained Natural Language Processing (NLP) model specifically designed for financial text, to perform sentiment analysis on the news articles.

- **Sentiment Categories**:  
  FinBERT categorized the sentiment of each news article into three classes:
  - Positive
  - Negative
  - Neutral

- **Additional Columns Added**:  
  The following columns were added to the news data after performing sentiment analysis:
  - `Sentiment`: Indicates the sentiment category (Positive, Negative, Neutral).
  - `Sentiment_Numerical`: A numerical representation of sentiment (e.g., 1 for Positive, 0 for Neutral, -1 for Negative).
  - `Sentiment_Score`: The confidence score for the sentiment prediction.

- **Purpose of Sentiment Analysis**:  
  By incorporating sentiment data, we aimed to:
  - Understand how market sentiment aligns with stock price trends.
  - Add a qualitative layer of insights to the historical stock data for better forecasting.


## Data Preparation and Cleaning

After collecting the data, we moved on to data cleaning and exploratory data analysis to prepare it for modeling. Here are the steps we followed:

1. **Trading Hours Filter**:  
   - We filtered the data to include only the time period from **9:30 AM to 4:00 PM**, as these are the standard trading hours for the stock market. This ensured that our analysis focused only on active trading periods. After filtering the dataset we left with 3.8 million records.

2. **Null Values Handling**:  
   - Checked the dataset for any null values and removed or imputed them as necessary.

3. **Dropped Unnecessary Columns**:  
   - We dropped the `Company Name` and `Number of Transactions` columns, as they were not relevant to our analysis or modeling.

4. **Combining Historical Data with News Sentiment**:  
   - The minute-level historical stock data (3.8 million records) was combined with the sentiment data from 50,000 news articles.  
   - **Logic for Combining**:  
     For each minute, the sentiment of the most recent news article was applied until a new article was published. This ensured that every minute had an associated sentiment value, even if there was no new article for that time period.

5. **Feature Engineering and Granularity**:  
   - We extracted **hour-level** and **day-level** data from the minute-level data and added a `Granularity` column to indicate the granularity for each record. This column helped the model differentiate between minute, hour, and day-level data.  
   - The news sentiment data was also aggregated to match the hour and day-level granularities, ensuring that sentiment information was available across all time scales. This allowed the dataset to capture both short-term and long-term patterns.

6. **Sorting Data**:  
   - The combined dataset, including historical stock data and news sentiment data, was sorted by `Timestamp` and `Stock Symbol` to ensure proper chronological order and stock-wise segmentation.

By performing these steps, we created a robust, multi-granularity dataset that incorporated news sentiment at minute, hour, and day levels. This powerful dataset provided the ability to capture both short-term and long-term patterns, making it highly effective for deep learning models focused on forecasting tasks.

