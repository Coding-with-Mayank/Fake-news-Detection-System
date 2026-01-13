import requests
from bs4 import BeautifulSoup
from newspaper import Article
import warnings
warnings.filterwarnings('ignore')

class ArticleScraper:
    """Enhanced web scraper for news articles"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def scrape_with_newspaper(self, url):
        """Scrape using newspaper3k library"""
        try:
            article = Article(url)
            article.download()
            article.parse()
            article.nlp()
            
            return {
                'title': article.title,
                'text': article.text,
                'authors': article.authors,
                'publish_date': str(article.publish_date) if article.publish_date else None,
                'summary': article.summary,
                'keywords': article.keywords,
                'source_url': url
            }
        except Exception as e:
            print(f"Newspaper3k error: {str(e)}")
            return None
    
    def scrape_with_beautifulsoup(self, url):
        """Fallback scraping with BeautifulSoup"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = soup.find('h1')
            title_text = title.get_text() if title else ""
            
            # Extract paragraphs
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs])
            
            return {
                'title': title_text,
                'text': text,
                'source_url': url
            }
        except Exception as e:
            return {'error': f"Error scraping URL: {str(e)}"}
    
    def scrape_article(self, url):
        """Main scraping method with fallback"""
        # Try newspaper3k first
        result = self.scrape_with_newspaper(url)
        
        if result and result.get('text'):
            return result
        
        # Fallback to BeautifulSoup
        return self.scrape_with_beautifulsoup(url)
