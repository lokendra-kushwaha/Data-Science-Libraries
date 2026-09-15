import pandas as pd
import ast
import re

def extract_names(text):
    """
    Safely evaluates a string representation of a list of dictionaries
    and extracts the 'name' values. Returns an empty list if data is missing/invalid.
    """
    # if the text is missing (NaN) or not a string, return empty list
    if not isinstance(text, str):
        return []
    
    try:
        names_list = []
        real_list = ast.literal_eval(text)
        
        for dictionary in real_list:
            names_list.append(dictionary['name'])
            
        return names_list
    except (ValueError, SyntaxError):
        return []


def remove_punctuation(text):
    """
    Removes all punctuation marks (commas, dots, quotes, etc.) from a string.
    Only keeps alphanumeric characters and spaces to ensure high accuracy for the Vectorizer.
    """
    if not isinstance(text, str):
        return ""
    
    # Regex: Find anything that is NOT a word character (\w) or a space (\s) and remove it.
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    return cleaned_text

STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'am', 'was', 'were',
    'be', 'been', 'being', 'in', 'on', 'at', 'to', 'for', 'with', 'about',
    'against', 'between', 'into', 'through', 'during', 'before', 'after',
    'above', 'below', 'from', 'up', 'down', 'out', 'off', 'over', 'under',
    'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where',
    'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
    'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than',
    'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now',
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your',
    'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she',
    'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
    'these', 'those', 'as', 'until', 'while', 'of', 'by', 'do', 'does', 'did',
    'doing', 'have', 'has', 'had', 'having', 'because', 'if'
}
def remove_stopwords(text: str) -> str:
    """
    Removes common English stop words to improve recommendation accuracy.
    """
    if not isinstance(text, str):
        return ""
    
    words = text.split()
    filtered_words = [word for word in words if word not in STOP_WORDS]
    return " ".join(filtered_words)

def prepare_movie_data(csv_file_path):
    """
    Reads, cleans, and merges movie data with HIGH MEMORY EFFICIENCY.
    Preserves the 'id' and 'title' columns for the recommendation engine UI.
    
    Args:
        csv_file_path (str): The path to the raw movies CSV file.
        
    Returns:
        pd.DataFrame: A cleaned DataFrame containing only 'id', 'title', and 'tags'.
    """
    print(f"⚙️ Loading and cleaning data from: {csv_file_path}...")

    required_cols = ['id', 'title', 'overview', 'genres', 'keywords']
    try:
        movies_df = pd.read_csv(csv_file_path, usecols=required_cols)
    except FileNotFoundError:
        print(f"❌ Error: The file '{csv_file_path}' was not found.")
        return pd.DataFrame()
    
    # 1. Extract lists from stringified JSON
    movies_df['genres'] = movies_df['genres'].apply(extract_names)
    movies_df['keywords'] = movies_df['keywords'].apply(extract_names)

    # 2. Convert string paragraphs to lists of words
    movies_df['overview'] = movies_df['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])
    movies_df['title_tag'] = movies_df['title'].apply(lambda x: x.split() if isinstance(x, str) else [])

    # 3. Concatenate all features into a single 'tags' list
    movies_df['tags'] = movies_df['title_tag'] + movies_df['overview'] + movies_df['genres'] + movies_df['keywords']

    # Destroy old columns immediately to free up RAM!
    movies_df.drop(columns=['overview', 'genres', 'keywords'], inplace=True)

    # 4. Final text processing on the newly created 'tags' column
    movies_df['tags'] = movies_df['tags'].apply(lambda x: " ".join(x).lower())
    movies_df['tags'] = movies_df['tags'].apply(remove_punctuation)
    movies_df['tags'] = movies_df['tags'].apply(remove_stopwords)
    
    print(f"✅ Data cleaning complete! Final shape: {movies_df.shape}")
    return movies_df


if __name__ == "__main__":
    # ==============================================================================
    # TESTING 
    # ==============================================================================
    print("Running tests for Data Cleaner Module...\n")
    
    fake_string_data = '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]'
    print("Before:", fake_string_data)
    print("After Extraction:", extract_names(fake_string_data))
    
    fake_dirty_text = 'Hello! "Avatar" is an epic, sci-fi movie.'
    print("\nBefore Cleaning:", fake_dirty_text)
    print("After Cleaning:", remove_punctuation(fake_dirty_text))

    