from flask import Flask, render_template, request, jsonify
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from textblob import TextBlob
import json
import os
from collections import defaultdict
import time
import pytz

app = Flask(__name__)

# Configuration
NEWS_API_KEY = "f425130bd2974416b8f24f7667396dd4"
ALPHA_VANTAGE_KEY = "7TQ1NR9P464ESNCA"

# Competitor mapping for suggestion system
COMPETITOR_MAP = {
    'AMZN': ['WMT', 'EBAY', 'SHOP', 'TGT'],  # Amazon competitors
    'AAPL': ['MSFT', 'GOOGL', 'SAMSUNG'],    # Apple competitors
    'TSLA': ['F', 'GM', 'NIO', 'RIVN'],      # Tesla competitors
    'GOOGL': ['MSFT', 'AAPL', 'META', 'AMZN'], # Google competitors
    'META': ['GOOGL', 'SNAP', 'TWTR', 'PINS'], # Meta competitors
    'MSFT': ['AAPL', 'GOOGL', 'ORCL', 'CRM'], # Microsoft competitors
    'NVDA': ['AMD', 'INTC', 'QCOM', 'TSM'],   # NVIDIA competitors
    'NFLX': ['DIS', 'ROKU', 'PARA', 'WBD'],   # Netflix competitors
    'WMT': ['AMZN', 'TGT', 'COST', 'HD'],     # Walmart competitors
    'JPM': ['BAC', 'WFC', 'C', 'GS'],         # JPMorgan competitors
}

# Company name to symbol mapping
COMPANY_SYMBOLS = {
    'amazon': 'AMZN', 'walmart': 'WMT', 'ebay': 'EBAY', 'shopify': 'SHOP', 'target': 'TGT',
    'apple': 'AAPL', 'microsoft': 'MSFT', 'google': 'GOOGL', 'samsung': 'SAMSUNG',
    'tesla': 'TSLA', 'ford': 'F', 'gm': 'GM', 'nio': 'NIO', 'rivian': 'RIVN',
    'meta': 'META', 'facebook': 'META', 'snap': 'SNAP', 'twitter': 'TWTR', 'pinterest': 'PINS',
    'nvidia': 'NVDA', 'amd': 'AMD', 'intel': 'INTC', 'qualcomm': 'QCOM',
    'netflix': 'NFLX', 'disney': 'DIS', 'roku': 'ROKU', 'paramount': 'PARA',
    'jpmorgan': 'JPM', 'bofa': 'BAC', 'wells fargo': 'WFC', 'citigroup': 'C',
}

class MarketStatusChecker:
    def __init__(self):
        self.market_holidays_2024 = [
            '2024-01-01', '2024-01-15', '2024-02-19', '2024-03-29',
            '2024-05-27', '2024-06-19', '2024-07-04', '2024-09-02',
            '2024-11-28', '2024-12-25'
        ]
        
    def is_market_open(self):
        """Check if US stock market is currently open"""
        try:
            # US Eastern Time
            eastern = pytz.timezone('US/Eastern')
            now_et = datetime.now(eastern)
            
            # Check if it's a weekend
            if now_et.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return False, "Market is closed (Weekend)"
            
            # Check if it's a holiday
            current_date = now_et.strftime('%Y-%m-%d')
            if current_date in self.market_holidays_2024:
                return False, "Market is closed (Holiday)"
            
            # Check market hours (9:30 AM - 4:00 PM ET)
            market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
            
            if market_open <= now_et <= market_close:
                return True, "Market is open"
            elif now_et < market_open:
                return False, f"Market opens at 9:30 AM ET (in {(market_open - now_et).seconds // 3600}h {((market_open - now_et).seconds % 3600) // 60}m)"
            else:
                next_open = market_open + timedelta(days=1)
                # Skip weekends
                while next_open.weekday() >= 5:
                    next_open += timedelta(days=1)
                return False, "Market is closed (After hours)"
                
        except Exception as e:
            return False, f"Unable to determine market status: {str(e)}"

class SentimentAnalyzer:
    def __init__(self):
        self.cache = {}
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using TextBlob"""
        if text in self.cache:
            return self.cache[text]

        try:
            if not text or len(text.strip()) < 10:
                self.cache[text] = ('neutral', 0.0)
                return 'neutral', 0.0

            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            sentiment = 'neutral'
            if polarity > 0.1:
                sentiment = 'positive'
            elif polarity < -0.1:
                sentiment = 'negative'
            
            self.cache[text] = (sentiment, polarity)
            return sentiment, polarity
        except Exception as e:
            print(f"Error analyzing sentiment for text: '{text[:50]}...' - {e}")
            self.cache[text] = ('neutral', 0.0)
            return 'neutral', 0.0

class NewsDataFetcher:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"
        self.cache = {}

    def fetch_news(self, query, days_back=7):
        """Fetch news articles from NewsAPI"""
        cache_key = f"{query}_{days_back}"
        if cache_key in self.cache and self.cache[cache_key]['timestamp'] > (datetime.now() - timedelta(hours=1)):
            print(f"Fetching news from cache for query: {query}, days_back: {days_back}")
            return self.cache[cache_key]['data']

        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            params = {
                'q': query,
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d'),
                'sortBy': 'relevancy',
                'apiKey': self.api_key,
                'language': 'en',
                'pageSize': 100
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            articles = []
            if data.get('status') == 'ok':
                for article in data.get('articles', []):
                    if not article.get('title') or not article.get('description'):
                        continue
                    articles.append({
                        'title': article.get('title', ''),
                        'description': article.get('description', ''),
                        'content': article.get('content', ''),
                        'published_at': article.get('publishedAt', ''),
                        'source': article.get('source', {}).get('name', ''),
                        'url': article.get('url', '')
                    })
            
            self.cache[cache_key] = {'data': articles, 'timestamp': datetime.now()}
            return articles
        except requests.exceptions.RequestException as e:
            print(f"Network or API error fetching news for '{query}': {e}")
            return []
        except json.JSONDecodeError:
            print(f"JSON decoding error for news API response for '{query}'")
            return []
        except Exception as e:
            print(f"An unexpected error occurred while fetching news for '{query}': {e}")
            return []

class StockDataFetcher:
    def __init__(self, alpha_vantage_key=None):
        self.alpha_vantage_key = alpha_vantage_key
        self.av_base_url = "https://www.alphavantage.co/query"
        self.yf_cache = {}
        self.av_cache = {}
        self.av_last_request_time = 0

    def _get_yfinance_data(self, symbol, period):
        """Internal helper to fetch stock price data using yfinance."""
        cache_key = f"yf_{symbol}_{period}"
        if cache_key in self.yf_cache and self.yf_cache[cache_key]['timestamp'] > (datetime.now() - timedelta(hours=1)):
            print(f"Fetching yfinance data from cache for symbol: {symbol}, period: {period}")
            return self.yf_cache[cache_key]['data']

        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                return []

            data = []
            for date, row in hist.iterrows():
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume'])
                })
            
            self.yf_cache[cache_key] = {'data': data, 'timestamp': datetime.now()}
            return data
        except Exception as e:
            print(f"Error fetching stock data from yfinance for {symbol} (period={period}): {e}")
            return []

    def _get_alpha_vantage_data(self, symbol, days_back):
        """Internal helper to fetch stock data from Alpha Vantage API."""
        if not self.alpha_vantage_key:
            print("Alpha Vantage API key not available")
            return []
        
        cache_key = f"av_daily_{symbol}_{days_back}"
        if cache_key in self.av_cache and self.av_cache[cache_key]['timestamp'] > (datetime.now() - timedelta(hours=1)):
            print(f"Fetching Alpha Vantage data from cache for symbol: {symbol}, days_back: {days_back}")
            return self.av_cache[cache_key]['data']

        if (time.time() - self.av_last_request_time) < 15:
            print("Alpha Vantage rate limit detected, waiting...")
            time.sleep(15 - (time.time() - self.av_last_request_time))
        
        try:
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'apikey': self.alpha_vantage_key,
                'outputsize': 'compact'
            }
            
            response = requests.get(self.av_base_url, params=params)
            response.raise_for_status()
            data = response.json()
            self.av_last_request_time = time.time()
            
            if 'Time Series (Daily)' in data:
                time_series = data['Time Series (Daily)']
                stock_data = []
                
                sorted_dates = sorted(time_series.keys(), reverse=True)
                
                count = 0
                for date in sorted_dates:
                    if count >= days_back:
                        break
                    day_data = time_series[date]
                    stock_data.append({
                        'date': date,
                        'open': float(day_data['1. open']),
                        'high': float(day_data['2. high']),
                        'low': float(day_data['3. low']),
                        'close': float(day_data['4. close']),
                        'volume': int(day_data['5. volume'])
                    })
                    count += 1
                
                stock_data.reverse()
                self.av_cache[cache_key] = {'data': stock_data, 'timestamp': datetime.now()}
                return stock_data
            else:
                error_message = data.get("Error Message", data.get("Note", "Unknown Alpha Vantage error"))
                print(f"Error in Alpha Vantage response for {symbol}: {error_message}")
                return []
                
        except requests.exceptions.RequestException as e:
            print(f"Network or API error fetching stock data from Alpha Vantage for {symbol}: {e}")
            return []
        except json.JSONDecodeError:
            print(f"JSON decoding error for Alpha Vantage API response for {symbol}")
            return []
        except Exception as e:
            print(f"An unexpected error occurred while fetching stock data from Alpha Vantage for {symbol}: {e}")
            return []

    def get_stock_data(self, symbol, days_back=7):
        """Fetch stock price data using yfinance as primary, Alpha Vantage as backup."""
        period = f"{days_back}d"
        data = self._get_yfinance_data(symbol, period)
        
        if not data:
            print(f"yfinance failed for {symbol}, trying Alpha Vantage...")
            data = self._get_alpha_vantage_data(symbol, days_back)
        
        return data
    
    def get_real_time_quote(self, symbol):
        """Get real-time quote from Alpha Vantage"""
        if not self.alpha_vantage_key:
            print("Alpha Vantage API key not available for real-time quote.")
            return {}

        cache_key = f"av_quote_{symbol}"
        if cache_key in self.av_cache and self.av_cache[cache_key]['timestamp'] > (datetime.now() - timedelta(minutes=5)):
            print(f"Fetching real-time quote from cache for symbol: {symbol}")
            return self.av_cache[cache_key]['data']

        if (time.time() - self.av_last_request_time) < 15:
            print("Alpha Vantage rate limit detected for quote, waiting...")
            time.sleep(15 - (time.time() - self.av_last_request_time))
            
        try:
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.alpha_vantage_key
            }
            
            response = requests.get(self.av_base_url, params=params)
            response.raise_for_status()
            data = response.json()
            self.av_last_request_time = time.time()
            
            if 'Global Quote' in data:
                quote = data['Global Quote']
                result = {
                    'symbol': quote.get('01. symbol', ''),
                    'price': float(quote.get('05. price', 0)),
                    'change': float(quote.get('09. change', 0)),
                    'change_percent': quote.get('10. change percent', '0%'),
                    'volume': int(quote.get('06. volume', 0)),
                    'latest_trading_day': quote.get('07. latest trading day', ''),
                    'previous_close': float(quote.get('08. previous close', 0))
                }
                self.av_cache[cache_key] = {'data': result, 'timestamp': datetime.now()}
                return result
            
            print(f"No Global Quote data found for {symbol}: {data.get('Error Message', 'Unknown response')}")
            return {}
        except requests.exceptions.RequestException as e:
            print(f"Network or API error fetching real-time quote for {symbol}: {e}")
            return {}
        except json.JSONDecodeError:
            print(f"JSON decoding error for real-time quote API response for {symbol}")
            return {}
        except Exception as e:
            print(f"An unexpected error occurred while fetching real-time quote for {symbol}: {e}")
            return {}
    
    def search_symbol(self, keywords):
        """Search for stock symbols using Alpha Vantage"""
        if not self.alpha_vantage_key:
            print("Alpha Vantage API key not available for symbol search.")
            return []

        cache_key = f"av_search_{keywords}"
        if cache_key in self.av_cache and self.av_cache[cache_key]['timestamp'] > (datetime.now() - timedelta(days=1)):
            print(f"Fetching symbol search from cache for keywords: {keywords}")
            return self.av_cache[cache_key]['data']

        if (time.time() - self.av_last_request_time) < 15:
            print("Alpha Vantage rate limit detected for search, waiting...")
            time.sleep(15 - (time.time() - self.av_last_request_time))

        try:
            params = {
                'function': 'SYMBOL_SEARCH',
                'keywords': keywords,
                'apikey': self.alpha_vantage_key
            }
            
            response = requests.get(self.av_base_url, params=params)
            response.raise_for_status()
            data = response.json()
            self.av_last_request_time = time.time()
            
            if 'bestMatches' in data:
                matches = []
                for match in data['bestMatches'][:10]:
                    matches.append({
                        'symbol': match.get('1. symbol', ''),
                        'name': match.get('2. name', ''),
                        'type': match.get('3. type', ''),
                        'region': match.get('4. region', ''),
                        'currency': match.get('8. currency', '')
                    })
                self.av_cache[cache_key] = {'data': matches, 'timestamp': datetime.now()}
                return matches
            
            print(f"No bestMatches found for keywords '{keywords}': {data.get('Error Message', 'Unknown response')}")
            return []
        except requests.exceptions.RequestException as e:
            print(f"Network or API error searching symbols for '{keywords}': {e}")
            return []
        except json.JSONDecodeError:
            print(f"JSON decoding error for symbol search API response for '{keywords}'")
            return []
        except Exception as e:
            print(f"An unexpected error occurred while searching symbols for '{keywords}': {e}")
            return []

class MarketSentimentAnalyzer:
    def __init__(self):
        self.news_fetcher = NewsDataFetcher(NEWS_API_KEY)
        self.stock_fetcher = StockDataFetcher(ALPHA_VANTAGE_KEY)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.market_status = MarketStatusChecker()
    
    def get_competitor_suggestions(self, symbol):
        """Get competitor suggestions for a given symbol"""
        competitors = COMPETITOR_MAP.get(symbol.upper(), [])
        suggestions = []
        
        # Get company names for popular assets
        popular_assets = [
            {'name': 'Tesla', 'symbol': 'TSLA'},
            {'name': 'Apple', 'symbol': 'AAPL'},
            {'name': 'NVIDIA', 'symbol': 'NVDA'},
            {'name': 'Amazon', 'symbol': 'AMZN'},
            {'name': 'Google', 'symbol': 'GOOGL'},
            {'name': 'Microsoft', 'symbol': 'MSFT'},
            {'name': 'Meta', 'symbol': 'META'},
            {'name': 'Netflix', 'symbol': 'NFLX'},
            {'name': 'Walmart', 'symbol': 'WMT'},
            {'name': 'eBay', 'symbol': 'EBAY'},
            {'name': 'Shopify', 'symbol': 'SHOP'},
            {'name': 'Target', 'symbol': 'TGT'},
            {'name': 'Ford', 'symbol': 'F'},
            {'name': 'GM', 'symbol': 'GM'},
            {'name': 'AMD', 'symbol': 'AMD'},
            {'name': 'Intel', 'symbol': 'INTC'},
            {'name': 'Disney', 'symbol': 'DIS'},
            {'name': 'Roku', 'symbol': 'ROKU'},
        ]
        
        # Create a mapping of symbols to names
        symbol_to_name = {asset['symbol']: asset['name'] for asset in popular_assets}
        
        for comp_symbol in competitors:
            name = symbol_to_name.get(comp_symbol, comp_symbol)
            suggestions.append({'name': name, 'symbol': comp_symbol})
        
        return suggestions
    
    def analyze_market_sentiment(self, asset, symbol, days_back=7):
        """Main function to analyze market sentiment"""
        # Get market status
        is_open, market_message = self.market_status.is_market_open()
        
        # Fetch news data
        news_articles = self.news_fetcher.fetch_news(asset, days_back)
        
        # Fetch stock data
        stock_data = self.stock_fetcher.get_stock_data(symbol, days_back)
        
        # Get real-time quote
        real_time_quote = self.stock_fetcher.get_real_time_quote(symbol)
        
        # Get competitor suggestions
        competitor_suggestions = self.get_competitor_suggestions(symbol)
        
        # Analyze sentiment for each article
        sentiment_data = []
        daily_sentiment = defaultdict(lambda: {'scores': [], 'articles': []})
        
        for article in news_articles:
            text = f"{article['title']} {article['description'] or ''}"
            sentiment, score = self.sentiment_analyzer.analyze_sentiment(text)
            
            article_date = article['published_at'][:10]
            
            sentiment_item = {
                'title': article['title'],
                'description': article['description'],
                'sentiment': sentiment,
                'score': score,
                'date': article_date,
                'source': article['source'],
                'url': article['url']
            }
            
            sentiment_data.append(sentiment_item)
            daily_sentiment[article_date]['scores'].append(score)
            daily_sentiment[article_date]['articles'].append(sentiment_item)
        
        # Calculate daily average sentiment
        daily_avg_sentiment = {}
        for date, data in daily_sentiment.items():
            scores = data['scores']
            daily_avg_sentiment[date] = {
                'avg_score': np.mean(scores) if scores else 0,
                'count': len(scores),
                'positive': len([s for s in scores if s > 0.1]),
                'negative': len([s for s in scores if s < -0.1]),
                'neutral': len([s for s in scores if -0.1 <= s <= 0.1])
            }
        
        # Generate insights
        insights = self.generate_insights(daily_avg_sentiment, stock_data)

        return {
            'sentiment_data': sentiment_data,
            'daily_sentiment': daily_avg_sentiment,
            'stock_data': stock_data,
            'real_time_quote': real_time_quote,
            'competitor_suggestions': competitor_suggestions,
            'market_status': {
                'is_open': is_open,
                'message': market_message,
                'current_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
            },
            'summary': {
                'total_articles': len(sentiment_data),
                'positive_count': len([s for s in sentiment_data if s['sentiment'] == 'positive']),
                'negative_count': len([s for s in sentiment_data if s['sentiment'] == 'negative']),
                'neutral_count': len([s for s in sentiment_data if s['sentiment'] == 'neutral'])
            },
            'insights': insights
        }
    
    def generate_insights(self, daily_sentiment, stock_data):
        insights = []
        
        sentiment_df = pd.DataFrame([
            {'date': pd.to_datetime(d), 'avg_score': v['avg_score']}
            for d, v in daily_sentiment.items()
        ]).set_index('date').sort_index()

        stock_df = pd.DataFrame(stock_data)
        if not stock_df.empty:
            stock_df['date'] = pd.to_datetime(stock_df['date'])
            stock_df = stock_df.set_index('date').sort_index()
            stock_df['daily_change_percent'] = stock_df['close'].pct_change() * 100
        else:
            insights.append("No stock data available to generate price-related insights.")
            return insights

        combined_df = sentiment_df.join(stock_df[['daily_change_percent']], how='inner')

        if combined_df.empty:
            insights.append("Insufficient overlapping data between sentiment and stock prices to generate insights.")
            return insights

        for date_str, row in combined_df.iterrows():
            date_display = date_str.strftime('%Y-%m-%d')
            score = row['avg_score']
            price_change = row['daily_change_percent']

            if pd.isna(price_change):
                continue

            sentiment_label = "neutral"
            if score > 0.3:
                sentiment_label = "strong positive"
            elif score > 0.1:
                sentiment_label = "positive"
            elif score < -0.3:
                sentiment_label = "strong negative"
            elif score < -0.1:
                sentiment_label = "negative"
            
            insight_text = None
            if sentiment_label == "strong positive" and price_change > 0.5:
                insight_text = f"On {date_display}, strong positive news sentiment coincided with a significant stock price increase."
            elif sentiment_label == "strong negative" and price_change < -0.5:
                insight_text = f"On {date_display}, strong negative news sentiment was reflected in a notable stock price decline."
            elif sentiment_label == "positive" and price_change < -0.5:
                insight_text = f"On {date_display}, despite positive sentiment, the stock price experienced a slight decrease."
            elif sentiment_label == "negative" and price_change > 0.5:
                insight_text = f"On {date_display}, despite negative sentiment, the stock price saw a slight increase."
            
            if insight_text and insight_text not in insights:
                insights.append(insight_text)
        
        if not insights:
            insights.append("No distinct sentiment-price correlations observed in the selected period.")

        return insights

# Initialize the analyzer
analyzer = MarketSentimentAnalyzer()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_sentiment():
    """API endpoint to analyze market sentiment"""
    try:
        data = request.get_json()
        asset = data.get('asset', 'Tesla')
        symbol = data.get('symbol', 'TSLA')
        days_back = int(data.get('days_back', 7))
        
        result = analyzer.analyze_market_sentiment(asset, symbol, days_back)
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in /api/analyze: {e}")
        return jsonify({'error': f'An internal server error occurred: {str(e)}'}), 500

@app.route('/api/quote/<symbol>')
def get_real_time_quote(symbol):
    """Get real-time quote for a symbol"""
    try:
        quote = analyzer.stock_fetcher.get_real_time_quote(symbol)
        return jsonify(quote)
    except Exception as e:
        print(f"Error in /api/quote/{symbol}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/search/<keywords>')
def search_symbols(keywords):
    """Search for stock symbols"""
    try:
        results = analyzer.stock_fetcher.search_symbol(keywords)
        return jsonify(results)
    except Exception as e:
        print(f"Error in /api/search/{keywords}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/competitors/<symbol>')
def get_competitors(symbol):
    """Get competitor suggestions for a symbol"""
    try:
        suggestions = analyzer.get_competitor_suggestions(symbol)
        return jsonify(suggestions)
    except Exception as e:
        print(f"Error in /api/competitors/{symbol}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/market-status')
def get_market_status():
    """Get current market status"""
    try:
        is_open, message = analyzer.market_status.is_market_open()
        return jsonify({
            'is_open': is_open,
            'message': message,
            'current_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
        })
    except Exception as e:
        print(f"Error in /api/market-status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/assets')
def get_popular_assets():
    """Get list of popular assets for quick selection"""
    assets = [
        {'name': 'Tesla', 'symbol': 'TSLA'},
        {'name': 'Apple', 'symbol': 'AAPL'},
        {'name': 'NVIDIA', 'symbol': 'NVDA'},
        {'name': 'Amazon', 'symbol': 'AMZN'},
        {'name': 'Google', 'symbol': 'GOOGL'},
        {'name': 'Microsoft', 'symbol': 'MSFT'},
        {'name': 'Meta', 'symbol': 'META'},
        {'name': 'Netflix', 'symbol': 'NFLX'},
        {'name': 'Walmart', 'symbol': 'WMT'},
        {'name': 'eBay', 'symbol': 'EBAY'},
        {'name': 'Shopify', 'symbol': 'SHOP'},
        {'name': 'Target', 'symbol': 'TGT'},
        {'name': 'Ford', 'symbol': 'F'},
        {'name': 'GM', 'symbol': 'GM'},
        {'name': 'AMD', 'symbol': 'AMD'},
        {'name': 'Intel', 'symbol': 'INTC'},
        {'name': 'Disney', 'symbol': 'DIS'},
        {'name': 'Roku', 'symbol': 'ROKU'},
        {'name': 'Bitcoin', 'symbol': 'BTC-USD'},
        {'name': 'Ethereum', 'symbol': 'ETH-USD'}
    ]
    return jsonify(assets)

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'news_api_configured': bool(NEWS_API_KEY),
        'alpha_vantage_api_configured': bool(ALPHA_VANTAGE_KEY),
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(debug=True)