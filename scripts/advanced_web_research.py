import requests
from bs4 import BeautifulSoup
import json
import re
from urllib.parse import urljoin, urlparse
import time
from typing import Dict, List, Any
import pandas as pd

class AdvancedWebResearcher:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.research_data = []
        
    def scrape_website(self, url: str) -> Dict[str, Any]:
        """Advanced web scraping with content analysis"""
        try:
            print(f"🔍 Scraping: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract comprehensive information
            data = {
                'url': url,
                'title': self.extract_title(soup),
                'meta_description': self.extract_meta_description(soup),
                'headings': self.extract_headings(soup),
                'paragraphs': self.extract_paragraphs(soup),
                'links': self.extract_links(soup, url),
                'images': self.extract_images(soup, url),
                'word_count': self.calculate_word_count(soup),
                'reading_time': self.estimate_reading_time(soup),
                'keywords': self.extract_keywords(soup),
                'sentiment': self.analyze_content_sentiment(soup),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            print(f"✅ Successfully scraped {url}")
            return data
            
        except Exception as e:
            print(f"❌ Error scraping {url}: {str(e)}")
            return {'url': url, 'error': str(e)}
    
    def extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title"""
        title_tag = soup.find('title')
        return title_tag.get_text().strip() if title_tag else "No title found"
    
    def extract_meta_description(self, soup: BeautifulSoup) -> str:
        """Extract meta description"""
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        return meta_desc.get('content', '').strip() if meta_desc else "No description found"
    
    def extract_headings(self, soup: BeautifulSoup) -> Dict[str, List[str]]:
        """Extract all headings (H1-H6)"""
        headings = {}
        for i in range(1, 7):
            h_tags = soup.find_all(f'h{i}')
            headings[f'h{i}'] = [tag.get_text().strip() for tag in h_tags]
        return headings
    
    def extract_paragraphs(self, soup: BeautifulSoup) -> List[str]:
        """Extract paragraph content"""
        paragraphs = soup.find_all('p')
        return [p.get_text().strip() for p in paragraphs if p.get_text().strip()]
    
    def extract_links(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract all links with analysis"""
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            full_url = urljoin(base_url, href)
            
            links.append({
                'text': link.get_text().strip(),
                'href': href,
                'full_url': full_url,
                'is_external': urlparse(full_url).netloc != urlparse(base_url).netloc
            })
        return links
    
    def extract_images(self, soup: BeautifulSoup, base_url: str) -> List[Dict[str, str]]:
        """Extract image information"""
        images = []
        for img in soup.find_all('img'):
            src = img.get('src', '')
            if src:
                full_url = urljoin(base_url, src)
                images.append({
                    'src': src,
                    'full_url': full_url,
                    'alt': img.get('alt', ''),
                    'title': img.get('title', '')
                })
        return images
    
    def calculate_word_count(self, soup: BeautifulSoup) -> int:
        """Calculate total word count"""
        text = soup.get_text()
        words = re.findall(r'\b\w+\b', text.lower())
        return len(words)
    
    def estimate_reading_time(self, soup: BeautifulSoup) -> str:
        """Estimate reading time (average 200 words per minute)"""
        word_count = self.calculate_word_count(soup)
        minutes = max(1, round(word_count / 200))
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    
    def extract_keywords(self, soup: BeautifulSoup, top_n: int = 10) -> List[Dict[str, Any]]:
        """Extract and rank keywords"""
        text = soup.get_text().lower()
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'}
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        word_freq = {}
        
        for word in words:
            if word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [{'keyword': word, 'frequency': freq} for word, freq in sorted_keywords[:top_n]]
    
    def analyze_content_sentiment(self, soup: BeautifulSoup) -> Dict[str, float]:
        """Basic sentiment analysis of content"""
        text = soup.get_text().lower()
        
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'happy', 'success', 'best', 'perfect', 'awesome', 'brilliant', 'outstanding']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'angry', 'frustrated', 'disappointed', 'worst', 'horrible', 'disgusting', 'annoying', 'stupid', 'useless']
        
        words = re.findall(r'\b\w+\b', text)
        total_words = len(words)
        
        if total_words == 0:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        positive_ratio = positive_count / total_words
        negative_ratio = negative_count / total_words
        neutral_ratio = 1 - positive_ratio - negative_ratio
        
        return {
            'positive': positive_ratio,
            'negative': negative_ratio,
            'neutral': max(0, neutral_ratio)
        }
    
    def research_topic(self, topic: str, urls: List[str] = None) -> Dict[str, Any]:
        """Comprehensive topic research"""
        print(f"🔬 Starting research on: {topic}")
        
        if urls is None:
            # In a real implementation, you would use search APIs
            urls = [
                "https://en.wikipedia.org/wiki/Artificial_intelligence",
                "https://example.com",  # Placeholder URLs
            ]
        
        research_results = {
            'topic': topic,
            'sources': [],
            'summary': {},
            'insights': [],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Scrape each URL
        for url in urls:
            try:
                scraped_data = self.scrape_website(url)
                if 'error' not in scraped_data:
                    research_results['sources'].append(scraped_data)
                    time.sleep(1)  # Be respectful to servers
            except Exception as e:
                print(f"⚠️ Skipping {url} due to error: {str(e)}")
        
        # Analyze collected data
        research_results['summary'] = self.analyze_research_data(research_results['sources'])
        research_results['insights'] = self.generate_research_insights(research_results['sources'], topic)
        
        self.research_data.append(research_results)
        return research_results
    
    def analyze_research_data(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze collected research data"""
        if not sources:
            return {}
        
        total_words = sum(source.get('word_count', 0) for source in sources)
        total_sources = len(sources)
        
        # Aggregate keywords
        all_keywords = {}
        for source in sources:
            for keyword_data in source.get('keywords', []):
                keyword = keyword_data['keyword']
                freq = keyword_data['frequency']
                all_keywords[keyword] = all_keywords.get(keyword, 0) + freq
        
        top_keywords = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)[:15]
        
        # Aggregate sentiment
        avg_sentiment = {
            'positive': sum(source.get('sentiment', {}).get('positive', 0) for source in sources) / total_sources,
            'negative': sum(source.get('sentiment', {}).get('negative', 0) for source in sources) / total_sources,
            'neutral': sum(source.get('sentiment', {}).get('neutral', 0) for source in sources) / total_sources
        }
        
        return {
            'total_sources': total_sources,
            'total_words': total_words,
            'average_words_per_source': total_words / total_sources if total_sources > 0 else 0,
            'top_keywords': [{'keyword': k, 'total_frequency': f} for k, f in top_keywords],
            'overall_sentiment': avg_sentiment,
            'reading_time_total': f"{max(1, round(total_words / 200))} minutes"
        }
    
    def generate_research_insights(self, sources: List[Dict[str, Any]], topic: str) -> List[str]:
        """Generate insights from research data"""
        insights = []
        
        if not sources:
            return ["No sources available for analysis"]
        
        # Source diversity
        domains = set()
        for source in sources:
            if 'url' in source:
                domain = urlparse(source['url']).netloc
                domains.add(domain)
        
        insights.append(f"📊 Analyzed {len(sources)} sources from {len(domains)} different domains")
        
        # Content analysis
        total_words = sum(source.get('word_count', 0) for source in sources)
        insights.append(f"📝 Total content analyzed: {total_words:,} words")
        
        # Sentiment analysis
        avg_positive = sum(source.get('sentiment', {}).get('positive', 0) for source in sources) / len(sources)
        if avg_positive > 0.1:
            insights.append(f"😊 Overall positive sentiment detected ({avg_positive:.1%})")
        elif avg_positive < 0.05:
            insights.append(f"😐 Neutral to negative sentiment in content")
        
        # Keyword insights
        all_keywords = {}
        for source in sources:
            for keyword_data in source.get('keywords', []):
                keyword = keyword_data['keyword']
                all_keywords[keyword] = all_keywords.get(keyword, 0) + 1
        
        if all_keywords:
            most_common = max(all_keywords.items(), key=lambda x: x[1])
            insights.append(f"🔑 Most frequently mentioned term: '{most_common[0]}' (appeared in {most_common[1]} sources)")
        
        # Content depth
        avg_headings = sum(len(source.get('headings', {}).get('h1', [])) + 
                          len(source.get('headings', {}).get('h2', [])) for source in sources) / len(sources)
        if avg_headings > 5:
            insights.append(f"📋 Well-structured content with average {avg_headings:.1f} main headings per source")
        
        return insights
    
    def export_research(self, filename: str = None) -> str:
        """Export research data to JSON"""
        if filename is None:
            filename = f"research_export_{int(time.time())}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.research_data, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Research data exported to {filename}")
        return filename
    
    def create_research_report(self, research_data: Dict[str, Any]) -> str:
        """Generate a formatted research report"""
        report = f"""
🔬 RESEARCH REPORT: {research_data['topic'].upper()}
{'=' * 60}

📅 Research Date: {research_data['timestamp']}
📊 Sources Analyzed: {research_data['summary'].get('total_sources', 0)}
📝 Total Words: {research_data['summary'].get('total_words', 0):,}
⏱️ Estimated Reading Time: {research_data['summary'].get('reading_time_total', 'N/A')}

💡 KEY INSIGHTS:
"""
        
        for i, insight in enumerate(research_data['insights'], 1):
            report += f"{i}. {insight}\n"
        
        report += f"""
🔑 TOP KEYWORDS:
"""
        
        for i, keyword_data in enumerate(research_data['summary'].get('top_keywords', [])[:10], 1):
            report += f"{i:2d}. {keyword_data['keyword']} (frequency: {keyword_data['total_frequency']})\n"
        
        sentiment = research_data['summary'].get('overall_sentiment', {})
        report += f"""
😊 SENTIMENT ANALYSIS:
   Positive: {sentiment.get('positive', 0):.1%}
   Negative: {sentiment.get('negative', 0):.1%}
   Neutral:  {sentiment.get('neutral', 0):.1%}

📚 SOURCES:
"""
        
        for i, source in enumerate(research_data['sources'], 1):
            report += f"{i:2d}. {source.get('title', 'Untitled')} - {source.get('url', 'No URL')}\n"
            report += f"    Words: {source.get('word_count', 0):,} | Reading Time: {source.get('reading_time', 'N/A')}\n"
        
        return report

# Demonstration
if __name__ == "__main__":
    researcher = AdvancedWebResearcher()
    
    print("🌐 Advanced Web Research System")
    print("=" * 50)
    
    # Example research
    topic = "Artificial Intelligence"
    urls = [
        "https://example.com",  # In real use, these would be actual URLs
        "https://httpbin.org/html"  # This is a test URL that returns HTML
    ]
    
    print(f"Starting research on: {topic}")
    research_results = researcher.research_topic(topic, urls)
    
    # Generate and display report
    report = researcher.create_research_report(research_results)
    print(report)
    
    # Export data
    export_file = researcher.export_research()
    print(f"\n✅ Research completed and exported to {export_file}")
