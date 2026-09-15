import numpy as np
import pandas as pd

def build_vocabulary(tags_list: list) -> tuple:
    """
    Extracts all unique words from a list of text strings to build a master vocabulary.
    
    Args:
        tags_list (List[str]): A list where each element is a paragraph of tags for a single movie.

    Returns:
        Tuple[List[str], Dict[str, int]]: A tuple containing:
            - master_vocab: A sorted list of all unique words across all movies.
            - word_to_index: A hash map dictionary for O(1) time complexity lookups.
    """
    print("🧠 Building the Master Vocabulary...")
    vocab_set = set()
    
    # 1. Extract unique words using a Set
    for sentence in tags_list:
        words = sentence.split()
        for word in words:
            vocab_set.add(word)
            
    # 2. Convert Set to a sorted List to freeze the indices
    master_vocab = sorted(list(vocab_set))
    
    # 3. Create a Hash Map for fast lookups
    word_to_index = {word: index for index, word in enumerate(master_vocab)}
    
    print(f"✅ Vocabulary built successfully! Total unique words: {len(master_vocab)}")
    return master_vocab, word_to_index


def create_vector_matrix(cleaned_data: pd.DataFrame) -> np.ndarray:
    """
    Converts text data into a Bag-of-Words mathematical matrix.
    
    Args:
        cleaned_data (pd.DataFrame): A cleaned DataFrame containing only 'id', 'title', and 'tags'.

    Returns:
        np.ndarray: A 2D NumPy array of shape (total_movies, total_words) containing word counts.
    """
    tags_list = cleaned_data['tags'].tolist()
    vocab, word_to_index = build_vocabulary(tags_list)
    total_words = len(vocab) # The total number of unique words (determines matrix columns)

    total_movies = len(tags_list)
    print(f"⚙️ Constructing a {total_movies} x {total_words} Vector Matrix...")
    
    # Initialize the matrix with 8-bit integers to save memory (RAM)
    vector_matrix = np.zeros((total_movies, total_words), dtype=np.uint8)
    
    # Fill the matrix
    for movie_index, sentence in enumerate(tags_list):
        words = sentence.split()
        for word in words:
            if word in word_to_index: 
                word_index = word_to_index[word]
                # Increment the count for that specific word in that specific movie
                vector_matrix[movie_index, word_index] += 1
                
    print(f"✅ Matrix constructed! Final shape: {vector_matrix.shape}")
    return vector_matrix