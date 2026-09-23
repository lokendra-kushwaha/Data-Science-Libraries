import numpy as np

def generate_transition_matrix(web_graph):
    """
    Converts a web graph dictionary into a NumPy transition matrix for PageRank.
    
    This function performs three main tasks:
    1. Indexes all unique URLs (both givers and receivers) to assign them a fixed row/col ID.
    2. Builds an N x N matrix where Columns = Voting Pages (Givers) and Rows = Target Pages (Receivers).
    3. Applies a mathematical patch to "dead-end" pages to prevent Rank Leakage.

    Args:
        web_graph (dict): A dictionary where keys are source URLs and values are lists of outbound URLs.

    Returns:
        tuple: 
            - M (numpy.ndarray): The N x N transition matrix.
            - unique_pages (list): The master list mapping matrix indices to actual URLs.
    """
    unique_pages = []

    # STEP 1: Build the Master Index
    # We must collect BOTH main pages and their links into a single list 
    # to ensure our matrix is perfectly square (N x N).
    for main_page, link_list in web_graph.items():
        if main_page not in unique_pages:
            unique_pages.append(main_page)
        for link in link_list:
            if link not in unique_pages:
                unique_pages.append(link)

    num_pages = len(unique_pages)
    
    # Initialize an N x N grid with 0.0 (Zero Matrix)
    M = np.zeros((num_pages, num_pages))

    # STEP 2: Distribute the Votes (Matrix Population)
    for main_page, link_list in web_graph.items():
        col_index = unique_pages.index(main_page) # Column acts as the Donor/Giver

        if len(link_list) == 0:
            continue

        # Split the page's authority equally among all its outbound links
        vote_weight = 1.0 / len(link_list)
        
        for link in link_list:
            row_index = unique_pages.index(link) # Row acts as the Receiver
            # Use += to accumulate votes in case a page links to the same target multiple times
            M[row_index, col_index] += vote_weight

    # STEP 3: The Master Dead-End Patch (Fixing the Black Holes)
    # Pages with no outbound links absorb rank but give nothing back, draining the system.
    # We fix this by forcing these empty columns to distribute their rank to everyone equally.
    
    col_sums = M.sum(axis=0)            # 1. Calculate the total weight distributed by each column
    dead_ends = (col_sums == 0)         # 2. Identify columns that gave out 0.0 total (The Black Holes)
    M[:, dead_ends] = 1.0 / num_pages   # 3. Fill these empty columns with uniform probabilities (1/N)
    
    return M, unique_pages


def calculate_pagerank(M, num_pages, damping_factor=0.85, max_iterations=100):
    """
    Runs the Power Iteration algorithm with convergence checking to calculate final PageRanks.
    
    Instead of a fixed number of loops, this function continuously updates the ranks until 
    the system stabilizes (converges). It dynamically stops early when the changes between 
    iterations are microscopic, saving computation time for small graphs and ensuring 
    high accuracy for large ones.

    Args:
        M (numpy.ndarray): The N x N transition matrix with the dead-end patch applied.
        num_pages (int): Total number of unique pages (N) in the network.
        damping_factor (float): The probability (0.85) that a user continues clicking links 
                                vs. randomly teleporting (0.15) to a new page.
        max_iterations (int): A safety limit to prevent infinite loops in massive datasets.

    Returns:
        numpy.ndarray: A 1D array of final PageRank scores where the sum equals exactly 1.0.
    """
    N = num_pages 
    
    # Initial state: Assume all pages start with exactly equal authority.
    # np.ones(N) / N ensures the total system authority always starts at 1.0 (100%),
    rank_vector = np.ones(N) / N 
    
    d = damping_factor
    counter = 0

    while counter < max_iterations:
        # Core PageRank Equation
        # 1. d * np.dot(M, rank_vector) -> 85% of rank flows naturally through matrix links
        # 2. (1 - d) / N -> 15% of rank is randomly teleported to all pages to prevent stalling
        new_rank = ((1 - d) / N) + (d * np.dot(M, rank_vector))

        # Convergence Check: 
        # If the difference between the old and new ranks is smaller than 0.000000000001 (1e-12),
        # the system has stabilized. We stop the loop immediately to save CPU time.
        if np.allclose(rank_vector, new_rank, atol=1e-12):
            return new_rank
        
        # Update the state for the next iteration
        rank_vector = new_rank
        counter += 1

    # Fallback return in case the model hits the iteration limit without converging
    return rank_vector

