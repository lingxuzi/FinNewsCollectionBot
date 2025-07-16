import dateutil.parser
import feedparser
import dateutil
import warnings
import os
import shutil

from concurrent.futures import ThreadPoolExecutor
from db.stock_query import StockQueryEngine
from tqdm import tqdm
from newspaper import Article
from diskcache import Deque

warnings.filterwarnings('ignore')

def parse_content(link):
    try:
        article = Article(link)
        article.download()
        article.parse()
        return article, link
    except:
        return None, link

def parse_rss(source_type, rss_path, workers):
    news = []
    feed = feedparser.parse(rss_path)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(parse_content, entry['link']): entry for entry in feed['entries']}
        for future in tqdm(futures):
            entry = futures[future]
            article, link = future.result()
            if article is not None:
                news.append({
                    'title': futures[future],
                    'content': article.text,
                    'date': dateutil.parser.parse(entry['published']),
                    'link': link,
                    'source_type': source_type
                })
    
    return news

def parse_rss_pool(config):
    shutil.rmtree(config['queue_path'], ignore_errors=True)
    os.makedirs(config['queue_path'], exist_ok=True)
    deque = Deque(config['queue_path'])
    news = []
    for source_type, rss_pool in config['rss_pool'].items():
        news.extend(parse_rss(source_type, rss_pool, config['workers']))
        if len(news) >= 50:
            deque.append(news)
            news = []
    if len(news) > 0:
        deque.append(news)
    
def update_news_to_db(config):
    queue = Deque(config['queue_path'])
    db = StockQueryEngine('10.26.0.8', '2000', 'hmcz', 'Hmcz_12345678')
    db.connect_async()

    while len(queue) > 0:
        try:
            news = queue.popleft()
            ret, e = db.insert_news(news)
            if ret:
                print('News insert success')
            else:
                print(f'News insert failed: {e}')
        except Exception as e:
            print(f'News insert failed: {e}')

def do_parse_news(config):
    parse_rss_pool(config)
    update_news_to_db(config)

        
    