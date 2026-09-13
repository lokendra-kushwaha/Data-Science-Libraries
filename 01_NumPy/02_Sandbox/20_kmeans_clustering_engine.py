import numpy as np

def k_means_engine(data_points, size, epochs=100):
    random_indices = np.random.choice(data_points.shape[0], size=size, replace=False)
    centroids = data_points[random_indices]

    def calculate_mean_points(centroids):
        delta = data_points[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        distances = np.sum(delta**2, axis=-1)
        labels = np.argmin(distances, axis=1)

        mean_points_list = []
        for i in range(size):
            nearest_points = data_points[labels == i]
            mean_point = np.mean(nearest_points, axis=0)
            mean_points_list.append(mean_point)

        return np.array(mean_points_list)

    new_centroids = calculate_mean_points(centroids)
    counter = 0
    while counter < epochs:
        mean_points = calculate_mean_points(new_centroids)
        
        if np.all(new_centroids == mean_points):
            print(f"AI Converged at epoch {counter}!")
            return mean_points
        
        new_centroids = mean_points
        counter += 1

    return new_centroids


import numpy as np

def k_means_engine(data_points, size, epochs=100):
    """
    Executes the K-Means clustering algorithm to find optimal cluster centroids.

    Args:
        data_points (np.ndarray): An N-dimensional array of shape (num_points, num_features) 
            representing the dataset to be clustered.
        size (int): The number of clusters (k) to partition the data into.
        epochs (int, optional): The maximum number of training iterations before 
            forcing the loop to stop. Defaults to 100.

    Returns:
        np.ndarray: An array of shape (size, num_features) containing the final 
        converged cluster centroids.
    """
    # Randomly select 'size' unique data points to serve as the initial centroids
    random_indices = np.random.choice(data_points.shape[0], size=size, replace=False)
    centroids = data_points[random_indices]

    def calculate_mean_points(centroids):
        """
        Assigns points to the nearest centroid and calculates the new geometric mean.

        Args:
            centroids (np.ndarray): An array of current cluster centers.

        Returns:
            np.ndarray: The updated cluster centers based on the mean of assigned points.
        """
        # Utilize broadcasting to compute Euclidean distances without Python loops
        # data_points shape becomes (num_points, 1, num_features)
        # centroids shape becomes (1, size, num_features)
        delta = data_points[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        distances = np.sum(delta**2, axis=-1)
        
        # Assign each data point to the closest centroid (returns an array of cluster IDs)
        labels = np.argmin(distances, axis=1)

        mean_points_list = []
        for i in range(size):
            # Apply boolean masking to isolate data points belonging to cluster 'i'
            nearest_points = data_points[labels == i]
            
            # Calculate the new center of gravity (mean) for this isolated cluster
            mean_point = np.mean(nearest_points, axis=0)
            mean_points_list.append(mean_point)

        return np.array(mean_points_list)

    # Perform the initial forward pass to kickstart the training loop
    new_centroids = calculate_mean_points(centroids)
    counter = 0
    
    # The main training loop (epochs)
    while counter < epochs:
        mean_points = calculate_mean_points(new_centroids)
        
        # Convergence check: If the centroids stop moving, the algorithm has found the optimal clusters
        if np.all(new_centroids == mean_points):
            print(f"AI Converged at epoch {counter}!")
            return mean_points
        
        # Update the centroids for the next iteration
        new_centroids = mean_points
        counter += 1

    # Return the latest centroids if the model hits the epoch limit before converging
    return new_centroids


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 1. Run the custom K-Means engine to find 10 centers
    data_points = np.random.uniform(-10, 10, (5000, 3)) 
    centroids = k_means_engine(data_points=data_points, size=10, epochs=1000)

    # 2. Map every data point to its final cluster (0 to 9) to assign colors
    delta = data_points[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    labels = np.argmin(np.sum(delta**2, axis=-1), axis=1)

    # 3. Set up the 3D figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 4. Plot the data points with 10 distinct colors (using 'tab10' colormap)
    ax.scatter(data_points[:, 0], data_points[:, 1], data_points[:, 2], 
            c=labels, cmap='tab10', s=10, alpha=0.6)

    # 5. Mark the 10 cluster centroids with a large red 'X'
    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], 
            c='red', marker='X', s=300, edgecolors='black', linewidths=2)

    plt.title("Dynamic 3D K-Means Clustering (10 Clusters)")
    plt.show()

if __name__ == "__main__":
    import time
    from sklearn.cluster import KMeans

    print("🔥 THE ULTIMATE K-MEANS SHOWDOWN 🔥")
    print("Generating 1,00,000 Data Points in 3D Space...\n")
    huge_data = np.random.uniform(-10, 10, (100000, 3))

    # 1. Racing Custom NumPy Engine
    print("⏳ Running Custom NumPy Engine...")
    start_time = time.time()
    custom_centroids = k_means_engine(data_points=huge_data, size=10, epochs=300)
    custom_time = time.time() - start_time
    print(f"✅ Custom Engine Time: {custom_time:.4f} seconds\n")

    # 2. Racing Scikit-Learn Engine
    print("⏳ Running Scikit-Learn Engine...")
    start_time = time.time()
    sklearn_model = KMeans(n_clusters=10, n_init='auto', max_iter=300) 
    sklearn_model.fit(huge_data)
    sklearn_time = time.time() - start_time
    print(f"✅ Scikit-Learn Time : {sklearn_time:.4f} seconds\n")

    # 3. The Verdict
    print("🏆 RESULTS 🏆")
    if custom_time < sklearn_time:
        print(f"🤯 MAGIC! custom NumPy engine BEAT Scikit-Learn by {sklearn_time / custom_time:.2f}x!")
    else:
        print(f"💼 Scikit-Learn won by {custom_time / sklearn_time:.2f}x.")