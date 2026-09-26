def search(query, inverted_index, unique_pages, ranks):
    """
    Searches the inverted index for a keyword and ranks the resulting URLs by their PageRank score.

    This function bridges Phase 1 (Data Extraction) and Phase 2 (Mathematical Authority). 
    It retrieves the raw set of URLs containing the keyword, maps each URL to its pre-calculated 
    PageRank score using the master index, and returns a sorted ranking.

    Args:
        query (str): The keyword to search for (e.g., "poetry", "einstein").
        inverted_index (dict): The dictionary mapping words to a set of URLs where they appear.
        unique_pages (list): The master list acting as the index (roll number) for all URLs.
        ranks (numpy.ndarray): The 1D array containing the final PageRank score for each URL.

    Returns:
        list: A list of tuples formatted as (URL, Score), sorted in descending order of Authority. 
              Returns an empty list if the query word is not found in the index.
    """
    query = query.lower()
    
    # 1. Look up the query in our dictionary
    if query not in inverted_index:
        return []
        
    matched_urls = inverted_index[query]
    search_results = []
    
    # 2. Attach the PageRank score to each matched URL (Authority Mapping)
    for url in matched_urls:
        # Ensure the URL exists in our master list to avoid index errors
        if url in unique_pages:
            url_index = unique_pages.index(url)
            score = ranks[url_index]
            
            # Bundle the URL and its score into a tuple and add it to our results list
            search_results.append((url, score))
            
    # 3. Sort the tuples to determine the final ranking
    # lambda x: x[1] tells the sort function to evaluate the 'score' (index 1) rather than the 'url' (index 0)
    search_results.sort(key=lambda x: x[1], reverse=True)
    
    return search_results


