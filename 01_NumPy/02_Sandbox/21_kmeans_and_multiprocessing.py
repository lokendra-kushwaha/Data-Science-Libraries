import numpy as np
import concurrent.futures
import time

# ---------------------------------------------------------
# STEP 1: THE MAPPER
# ---------------------------------------------------------
def __compute_partial_sums(data_chunk, centroids):
    delta = data_chunk[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    labels = np.argmin(np.sum(delta**2, axis=-1), axis=1)
    
    k = centroids.shape[0]
    partial_sums = np.zeros_like(centroids) 
    partial_counts = np.zeros(k)
    
    for i in range(k):
        cluster_points = data_chunk[labels == i]
        if len(cluster_points) > 0:
            partial_sums[i] = np.sum(cluster_points, axis=0) 
            partial_counts[i] = len(cluster_points)       
            
    return partial_sums, partial_counts

# ---------------------------------------------------------
# STEP 2: THE REDUCER (The Main Engine)
# ---------------------------------------------------------
def kmeans_multiprocessing(data_points, size=10, epochs=100, num_cores=4):
    centroids = data_points[np.random.choice(data_points.shape[0], size, replace=False)]
    
    data_chunks = np.array_split(data_points, num_cores)
    
    counter = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
        while counter < epochs:
            centroids_list = [centroids] * len(data_chunks)
            results = list(executor.map(__compute_partial_sums, data_chunks, centroids_list))
            
            total_sums = np.sum([res[0] for res in results], axis=0)
            total_counts = np.sum([res[1] for res in results], axis=0)
            
            total_counts[total_counts == 0] = 1 
            new_centroids = total_sums / total_counts[:, np.newaxis]
            
            if np.all(centroids == new_centroids):
                print(f"⚡ AI Converged via Multi-Processing at epoch {counter}!")
                return new_centroids
                
            centroids = new_centroids
            counter += 1
            
    return centroids

if __name__ == '__main__':
    print("K-Means Engine...")
    huge_data = np.random.uniform(-10, 10, (100000, 3))
    
    start = time.time()
    final_centers = kmeans_multiprocessing(huge_data, size=10, epochs=300, num_cores=4)
    print(f"Time Taken: {time.time() - start:.4f} seconds")

if __name__ == "__main__":
    import time
    from sklearn.cluster import KMeans

    print("🔥 THE ULTIMATE K-MEANS SHOWDOWN 🔥")
    print("Generating 1,00,000 Data Points in 3D Space...\n")
    huge_data = np.random.uniform(-10, 10, (100000, 3))

    # 1. Racing Custom NumPy Multiprocessing Engine
    print("⏳ Running Custom NumPy Multiprocessing Engine...")
    start_time = time.time()
    custom_centroids = kmeans_multiprocessing(data_points=huge_data, size=10, epochs=300)
    custom_time = time.time() - start_time
    print(f"✅ Custom Engine Time: {custom_time:.4f} seconds\n")

    # 2. Racing Scikit-Learn
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