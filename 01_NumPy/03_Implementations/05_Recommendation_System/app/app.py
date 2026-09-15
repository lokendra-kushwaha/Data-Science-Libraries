import sys
import os

# 1. Path Fix for Streamlit Cloud
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))

# 2. Forcefully adding directories to Python's VIP search list
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
    /* Styling for the main title */
    .main-title {
        font-size: 3.5rem;
        color: #E50914; /* Netflix Red */
        text-align: center;
        font-weight: 800;
        margin-bottom: 0px;
    }
    
    /* Styling for the subtitle */
    .sub-title {
        text-align: center;
        font-size: 1.2rem;
        color: #888888;
        margin-bottom: 30px;
    }
    
    /* Styling for the movie names above posters */
    .movie-title {
        font-size: 1.1rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
        color: #1E1E1E;
        min-height: 55px; /* Keeps posters aligned even if movie names are long */
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
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
st.markdown("<h1 class='main-title'>Movie Recommendation Engine 🍿</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Discover your next favorite movie based on your tastes!</p>", unsafe_allow_html=True)

# Load the saved pickle data files
movies_file_path = os.path.join(current_dir, 'movies.pkl')
matrix_file_path = os.path.join(current_dir, 'matrix.pkl')

movies = pickle.load(open(movies_file_path, 'rb'))
matrix = pickle.load(open(matrix_file_path, 'rb'))

movie_names = movies['title'].values

# Create a searchable selectbox for the user
selected_movie = st.selectbox(
    "Select or type a movie you like:",
    movie_names
)

# Trigger action when the user clicks the button
if st.button("Get Recommendations"):
    
    recs = get_recommendations(selected_movie, cleaned_df=movies, vector_matrix=matrix)
    
    if recs:
        st.success(f"Because you liked '{selected_movie}', we recommend:")
        
        # Create 5 columns to display posters side by side
        cols = st.columns(5)
        
        # Loop through the columns and recommendations
        for i in range(5):
            with cols[i]:
                # Display the beautifully styled movie name
                st.markdown(f"<div class='movie-title'>{recs[i]}</div>", unsafe_allow_html=True)
                # Display the poster (use_column_width makes it fit perfectly)
                st.image(fetch_poster(recs[i]), use_container_width=True)
                
    else:
        st.error("Sorry, we couldn't find any recommendations for this movie.")

st.markdown("<div class='footer'>Developed with ❤️ by Lokendra Kushwaha</div>", unsafe_allow_html=True)

