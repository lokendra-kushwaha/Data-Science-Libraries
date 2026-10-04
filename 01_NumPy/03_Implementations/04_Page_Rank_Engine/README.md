# Modular PageRank & Search Engine

A fully modular, NumPy-powered web search engine built from scratch in Python. This project implements the core algorithms that drive the modern web: Breadth-First Search (BFS) for crawling, Inverted Indices for text retrieval, and the PageRank mathematical model (Power Iteration) for global authority scoring.

## 🏗️ System Architecture

The engine is separated into three distinct pipelines to ensure scalability, connected via a central Jupyter Notebook (`testing.ipynb`).

| Module | Role | Core Technologies |
| :--- | :--- | :--- |
| `crawler.py` | The Collector | `requests`, `BeautifulSoup`, `urllib.parse` |
| `page_rank.py` | The Math Brain | `numpy` (Transition Matrices, Vectors) |
| `search_engine.py` | The Matchmaker | Python Dictionaries, Lambda Sorting |

## ⚙️ Data Flow & Core Logic

**Phase 1: Data Ingestion (Crawler)**
* **Web Graph Generation:** Utilizes a BFS algorithm to navigate through seed URLs, collecting outbound links while maintaining a strict `visited` queue to prevent cyclic infinite loops.
* **Inverted Indexing:** Parses raw HTML into simple text, mapping every unique word to a Python `Set` of URLs. This ensures lightning-fast keyword lookups without duplicate entries.

**Phase 2: Authority Scoring (PageRank)**
* **Matrix Construction:** Transforms the web graph dictionary into an N x N NumPy transition matrix. 
* **The Dead-End Patch:** Identifies "black hole" pages (pages with 0 outbound links) using `M.sum(axis=0)` and redistributes their authority uniformly to prevent rank leakage.
* **Power Iteration & Early Stopping:** Simulates the random surfer model using a Damping Factor (0.85). Instead of a fixed loop, it dynamically converges using `np.allclose(atol=1e-6)`, stopping calculations early once scores stabilize to save CPU cycles.

**Phase 3: The Search Interface (Matchmaker)**
* Cross-references the user's query against the Inverted Index to find relevant pages.
* Maps the resulting URLs to their pre-computed global PageRank scores using a master ID index.
* Returns a sorted list of tuple pairs `(URL, Score)` in descending order of true authority.

## 🚀 Usage

Run the complete pipeline directly from your main notebook:

```python
# 1. Crawl Seed Domains
my_seeds = ["https://books.toscrape.com/", "https://quotes.toscrape.com/"]
graph, inverted_index = build_web_graph(my_seeds, max_pages=150)

# 2. Build Matrix & Calculate Authority
matrix, unique_pages = generate_transition_matrix(graph)
ranks = calculate_pagerank(matrix, len(unique_pages))

# 3. Query the Engine
results = search("poetry", inverted_index, unique_pages, ranks)
```

---