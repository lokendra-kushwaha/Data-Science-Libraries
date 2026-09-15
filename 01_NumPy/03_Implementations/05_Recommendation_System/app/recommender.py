import numpy as np
import pandas as pd
import difflib

def get_recommendations(movie_name: str, cleaned_df: pd.DataFrame, vector_matrix: np.ndarray, top_n: int = 5) -> list:
    """
    Finds the closest matching movie title and calculates cosine similarity
    to recommend the top N similar movies.

    Args:
        movie_name (str): The name of the movie searched by the user.
        cleaned_df (pd.DataFrame): The cleaned Pandas DataFrame containing 'title' column.
        vector_matrix (np.ndarray): The 2D NumPy array representing the Bag-of-Words vectors.
        top_n (int, optional): Number of recommendations to return. Defaults to 5.

    Returns:
        Optional[List[str]]: A list of recommended movie titles, or None if no match is found.
    """
    # 1. Title Matching (Search Engine)
    all_titles = cleaned_df['title'].tolist()
    matches = difflib.get_close_matches(movie_name, all_titles, n=1, cutoff=0.5)
    
    if not matches:
        return [f"\n❌ Sorry, no movie found matching '{movie_name}'. Try another spelling!"]
        
    matched_title = matches[0]
    print(f"\n✅ Did you mean: {matched_title}?")
    
    # 2. Get the Index of the matched movie
    movie_index = cleaned_df[cleaned_df['title'] == matched_title].index[0]
    
    # 3. Extract the target vector
    target_vector = vector_matrix[movie_index]
    
    # 4. Cosine Similarity Math
    dot_product = np.dot(vector_matrix, target_vector)
    norm_target = np.linalg.norm(target_vector)
    norm_all_movies = np.linalg.norm(vector_matrix, axis=1)
    
    # Prevent division by zero if an empty vector exists
    denominator = norm_target * norm_all_movies
    denominator = np.where(denominator == 0, 1e-9, denominator) 
    
    similarity_scores = dot_product / denominator
    
    # 5. Pair scores with original indices and Sort
    scores_with_index = list(enumerate(similarity_scores))
    sorted_scores = sorted(scores_with_index, reverse=True, key=lambda x: x[1])
    
    # 6. Extract Top N Recommendations
    top_movies_data = sorted_scores[1:top_n+1]
    
    recommended_titles = []
    
    for item in top_movies_data:
        idx = item[0] 
        movie_title = cleaned_df.iloc[idx]['title']
        recommended_titles.append(movie_title)
        
    return recommended_titles