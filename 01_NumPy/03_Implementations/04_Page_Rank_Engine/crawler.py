import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urldefrag

def build_web_graph(seed_urls, max_pages=10):
    """
    Crawls a list of seed URLs and builds a unified web graph and inverted index.

    Args:
        seed_urls (list): A list of starting URLs (e.g., ["https://site1.com", "https://site2.com"]).
        max_pages (int): The total maximum number of pages to crawl across ALL provided URLs.

    Returns:
        tuple: (web_graph (dict), inverted_index (dict)) containing combined data.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    }
    
    # Initialize the queue with all provided seed URLs
    to_visit = seed_urls.copy()  
    visited = set()        
    web_graph = {}
    inverted_index = {}
    pages_crawled = 1
    
    while to_visit and len(visited) < max_pages:
        current_url = to_visit.pop(0) 
        
        if current_url in visited:
            continue
            
        web_graph[current_url] = []
            
        print(f"🌐 Crawling [{pages_crawled}/{max_pages}]: {current_url}")

        try:
            response = requests.get(current_url, headers=headers, timeout=5)
            if response.status_code != 200:
                print(f"⚠️ Blocked/Error! Status code: {response.status_code}")
                continue
        except requests.RequestException:
            print(f"⚠️ Failed to connect to {current_url}")
            continue
            
        visited.add(current_url)
        soup = BeautifulSoup(response.text, 'html.parser')
    
        # 1. Building Inverted Index
        # Extract text, convert to lowercase, and split into simple words
        page_text = soup.get_text().lower()
        words = page_text.split()
        
        for word in words:
            if word not in inverted_index:
                inverted_index[word] = set()
            inverted_index[word].add(current_url)

        # 2. Building Web Graph
        # The href=True filter guarantees that every tag in this loop HAS an href attribute.
        for link in soup.find_all('a', href=True):
            # Beautiful Soup parses the HTML tag into a Tag Object. 
            # This object behaves exactly like a Python dictionary, allowing us to extract 
            # the value of HTML attributes (like 'href', 'title', or 'class') using keys.
            href = link['href']

            # urljoin combines the base domain with the relative path
            full_link = urljoin(current_url, href)
            full_link, _ = urldefrag(full_link)
            
            # Filter to ensure only queue valid web links (ignore javascript:, mailto:, etc.)
            if full_link.startswith('http'):
                if full_link not in visited and full_link not in to_visit:
                    to_visit.append(full_link)
                    web_graph[current_url].append(full_link)

        pages_crawled += 1

    return web_graph, inverted_index
