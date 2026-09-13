import numpy as np

def page_rank_engine():
    """
    Calculates the PageRank of a simplified web network using matrix multiplication.
    
    This function simulates the 'Random Surfer Model' by iteratively updating
    the authority (rank) of each page based on the incoming links from other pages.
    The iteration stops once the rank distribution stabilizes (converges).
    
    Returns:
        np.ndarray: A 1D array containing the final PageRank probabilities for each page.
    """
    # Transition Matrix M (Columns = Outgoing votes, Rows = Incoming votes)
    #                 A      B      C      D
    M = np.array([[0.00,  0.00,  1.00,  0.00],  # Links received by Page A
                  [ 1/3,  0.00,  0.00,  0.50],  # Links received by Page B
                  [ 1/3,  0.00,  0.00,  0.50],  # Links received by Page C
                  [ 1/3,  0.00,  0.00,  0.00]]) # Links received by Page D
                
    # Initial state: Assume all 4 pages start with equal authority (25% or 0.25)
    rank_vector = np.array([0.25, 0.25, 0.25, 0.25])
    N = len(rank_vector)  # Total number of pages (here's 4)
    d = 0.85 # The Google Damping Factor
    
    counter = 0
    max_iterations = 10000
    
    # The iteration loop (capped at max_iterations to prevent infinite execution)
    while counter < max_iterations:
        # Calculate the new authority distribution using the dot product
        # Note: np.dot automatically handles the 1D array transposition internally
        # 1. d * np.dot(M, rank_vector) -> 85% of the rank comes via links
        # 2. (1 - d) / N -> 15% of the rank is freely distributed to every page via 'teleportation'
        # new_rank = np.dot(M, rank_vector)
        new_rank = ((1 - d) / N) + (d * np.dot(M, rank_vector))

        # Convergence Check: Stop if the difference is smaller than 0.000001 (1e-6)
        # This handles floating-point precision issues in Python
        if np.allclose(rank_vector, new_rank, atol=1e-6):
            print(f"🔥 AI Converged in just {counter} iterations!")
            return new_rank
        
        # Update the rank vector for the next iteration
        rank_vector = new_rank
        counter += 1

    # Fallback return in case the model hits the iteration limit without converging
    print("Warning: Max iterations reached without convergence.")
    return rank_vector

# Execute the engine and print the final distribution
final_ranks = page_rank_engine()
print("Final PageRanks:", final_ranks)
