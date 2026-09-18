import numpy as np

def calculate_cosine_similarity(vec_a, vec_b):

    dot_product = np.dot(vec_a, vec_b)
    
    norm_a = np.linalg.norm(vec_a) # Magnitude of vec_a
    norm_b = np.linalg.norm(vec_b) # Magnitude of vec_b
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
        
    return dot_product / (norm_a * norm_b)

# ==============================================================================
# 🎬 THE DATASET (Rows = Users, Columns = Movies)
# Movies: [Avengers (Action), John Wick (Action), Hera Pheri (Comedy), Dhamaal (Comedy)]
# ==============================================================================

database_users = {
    "Rahul (Action Fan)": np.array([5, 4, 1, 0]),
    "Priya (Comedy Fan)": np.array([1, 0, 5, 4]),
    "Amit (Balanced)":    np.array([3, 3, 3, 3])
}

new_user_vector = np.array([5, 5, 0, 0])

# ==============================================================================
# 🚀 THE RECOMMENDATION LOGIC (Finding the closest match)
# ==============================================================================
print("🔍 Searching for the most similar user...\n")

best_match = None
highest_similarity = -1.0 # Starting with the lowest possible angle (-1)

for user_name, old_user_vector in database_users.items():
    
    similarity_score = calculate_cosine_similarity(new_user_vector, old_user_vector)

    # Converting to percentage for reading
    match_percentage = round(similarity_score * 100, 2)
    print(f"Angle match with {user_name}: {match_percentage}%")
    
    # Finding the max (costheta closest to 1)
    if similarity_score > highest_similarity:
        highest_similarity = similarity_score
        best_match = user_name

print("-" * 40)
print(f"🎯 RESULT: The new user is most similar to '{best_match}'")