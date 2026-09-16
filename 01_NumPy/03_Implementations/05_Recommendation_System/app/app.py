import sys
import os
import difflib

# 1. Path Fix for Streamlit Cloud
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

# 2. Forcefully adding directories to Python's search list
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

import streamlit as st
import pickle
import streamlit as st
import pickle
import requests
from recommender import get_recommendations

# Set page configuration for a wider layout
st.set_page_config(page_title="Movie Recommender", layout="wide")

# Inject Custom CSS for styling the web app
st.markdown("""
    <style>
    /* Styling for the footer with your name */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f3f6;
        color: #333;
        text-align: center;
        padding: 12px;
        font-size: 1.1rem;
        font-weight: bold;
        border-top: 3px solid #E50914;
        z-index: 100;
    }

    .movie-card {
        text-align: center;
        border-radius: 12px;
        padding: 8px;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        background-color: transparent;
    }

    .movie-card:hover {
        transform: translateY(-10px); /* Moves the card up slightly */
    }

    .movie-poster {
        width: 100%;
        border-radius: 12px; /* Rounded corners for the image */
        box-shadow: 0 6px 12px rgba(0,0,0,0.2); /* Soft shadow */
        margin-bottom: 10px;
    }

    .movie-title-card {
        font-size: 20px;
        font-weight: 800;
        color: #1f2937; /* Premium Dark Gray */
        margin-top: 12px;
        line-height: 1.4;
        letter-spacing: 0.3px;
        
        /* : Keeps all cards same height and adds '...' for long names */
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        text-overflow: ellipsis;
        min-height: 42px; /* Space reserved for exactly 2 lines */
        }
    </style>
""", unsafe_allow_html=True)

# Function to fetch movie poster from OMDb API
def fetch_poster(movie_title):
    url = f"https://www.omdbapi.com/?t={movie_title}&apikey=c5bf42ab"
    response = requests.get(url)
    data = response.json()
    
    if 'Poster' in data and data['Poster'] != 'N/A':
        return data['Poster']
    else:
        return "https://dummyimage.com/300x450/cccccc/000000.png&text=No+Poster"

# Display the styled Title and Subtitle
st.markdown(
    "<div style='text-align: center; font-weight: 800; font-size: clamp(1.2rem, 5vw, 3rem); word-break: keep-all; padding-bottom: 0.2rem; color: #E50914;'>Movie Recommendation Engine 🍿</div>", 
    unsafe_allow_html=True
)

st.markdown(
    "<div style='text-align: center; color: #666; font-size: clamp(0.9rem, 3vw, 1.2rem); margin-bottom: 2rem;'>Discover your next favorite movie based on your tastes!</div>", 
    unsafe_allow_html=True
)

# Load the saved pickle data files
movies_file_path = os.path.join(current_dir, 'movies.pkl')
matrix_file_path = os.path.join(current_dir, 'matrix.pkl')

movies = pickle.load(open(movies_file_path, 'rb'))
matrix = pickle.load(open(matrix_file_path, 'rb'))

movie_names = movies['title'].values

# Create special options for the dropdown
default_option = "🎬 Choose a movie from the dropdown..."
custom_option = "🔍 Cannot find it? Type your own spelling here..."

# Add both special options to the top of the list
options_list = [default_option, custom_option] + list(movie_names)

# 1. Primary Dropdown Menu (Defaults to index 0: 'Choose a movie...')
selected_option = st.selectbox(
    "Select a movie from the list or type your own:",
    options_list,
    index=0
)

# 2. Conditional Text Box
if selected_option == custom_option:
    final_movie = st.text_input(
        "Type the movie name (Don't worry about spelling!):",
        placeholder="e.g. Avatabne, Titanc..."
    )
elif selected_option == default_option:
    final_movie = ""  # Keep it empty so the landing page triggers
else:
    final_movie = selected_option


# 3. Auto-Trigger Logic (Runs when a movie is selected/searched)
if final_movie:
    
    # Find the closest matching movie in our database
    close_matches = difflib.get_close_matches(final_movie, movie_names, n=1)
    
    if close_matches:
        corrected_movie = close_matches[0]
        
        # Check if the spelling was exact or if AI corrected it
        if final_movie.lower() == corrected_movie.lower():
            # Perfect spelling
            display_message = f"Because you liked '{corrected_movie}', we recommend:"
            bg_color = "#d4edda" # Light Green background
            text_color = "#155724" # Dark Green text
        else:
            # Spelling was wrong, AI corrected it
            display_message = f"💡 Did you mean '{corrected_movie}'? Here are similar movies:"
            bg_color = "#fff3cd" # Light Yellow background
            text_color = "#856404" # Dark Yellow/Brown text
            
        # Display the dynamic message box
        st.markdown(
            f"<div style='text-align: center; color: {text_color}; font-weight: 600; padding: 12px; background-color: {bg_color}; border-radius: 8px; margin-bottom: 25px; font-size: clamp(0.9rem, 2.5vw, 1.1rem);'>{display_message}</div>", 
            unsafe_allow_html=True
        )
        
        # Fetch recommendations using the CORRECTED movie name
        recs = get_recommendations(corrected_movie, cleaned_df=movies, vector_matrix=matrix)
        
        if recs:
            # Create 5 columns to display posters side by side
            cols = st.columns(5)
            
            # Loop through the columns and recommendations
            for i in range(5):
                with cols[i]:
                    poster_url = fetch_poster(recs[i])
                    movie_name = recs[i]
                    
                    card_html = f"""
                        <div class="movie-card">
                            <img src="{poster_url}" class="movie-poster" alt="{movie_name}">
                            <div class="movie-title-card">{movie_name}</div>
                        </div>
                    """
                    st.markdown(card_html, unsafe_allow_html=True)
                    
    else:
        # If the spelling is completely unrecognizable
        st.error(f"Sorry, we couldn't find any match for '{final_movie}'. Please try another spelling!")


# 4. DEFAULT LANDING PAGE (Runs when no movie is searched yet)
else:
    # Welcome message
    st.markdown(
        "<div style='text-align: center; color: #0c5460; font-weight: 600; padding: 12px; background-color: #d1ecf1; border-radius: 8px; margin-bottom: 25px; font-size: clamp(0.9rem, 2.5vw, 1.1rem);'>🔥 Popular Picks to Get You Started:</div>", 
        unsafe_allow_html=True
    )
    
    # Top 10 default movies to display (2 rows of 5)
    default_movies = [
        "Inception", "The Dark Knight", "Interstellar", "The Matrix", "The Avengers",
        "Titanic", "Avatar", "The Shawshank Redemption", "Jurassic Park", "Spider-Man"
    ]
    
    # --- ROW 1 (First 5 Movies) ---
    cols1 = st.columns(5)
    for i in range(5):
        with cols1[i]:
            poster_url = fetch_poster(default_movies[i])
            movie_name = default_movies[i]
            
            card_html = f"""
                <div class="movie-card">
                    <img src="{poster_url}" class="movie-poster" alt="{movie_name}">
                    <div class="movie-title-card">{movie_name}</div>
                </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- ROW 2 (Next 5 Movies) ---
    cols2 = st.columns(5)
    for i in range(5, 10):
        with cols2[i - 5]:
            poster_url = fetch_poster(default_movies[i])
            movie_name = default_movies[i]
            
            card_html = f"""
                <div class="movie-card">
                    <img src="{poster_url}" class="movie-poster" alt="{movie_name}">
                    <div class="movie-title-card">{movie_name}</div>
                </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer' style='text-align:center; margin-top:50px;'>Developed with ❤️ by Lokendra Kushwaha</div>", unsafe_allow_html=True)