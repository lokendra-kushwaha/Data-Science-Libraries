import numpy as np
import time

class NBodySimulator3D:
    """
    A True 3D N-Body Gravity Simulator (X, Y, Z coordinates).
    """
    def __init__(self, num_bodies, G=1.0, softening=0.1):
        self.N = num_bodies
        self.G = G
        self.softening = softening # Prevents division by zero when bodies crash
        
        # Initialize Random Positions (X, Y, Z) and Velocities (Vx, Vy, Vz)
        self.positions = np.random.uniform(-100, 100, (self.N, 3))
        self.velocities = np.random.uniform(-1, 1, (self.N, 3))
        self.masses = np.random.uniform(1, 10, (self.N, 1))

    def compute_forces_python(self):
        """O(N^2) Pure Python implementation for 3D."""
        forces = np.zeros((self.N, 3))
        
        for i in range(self.N):
            for j in range(self.N):
                if i == j: continue
                
                # Vector distance
                dx = self.positions[j, 0] - self.positions[i, 0]
                dy = self.positions[j, 1] - self.positions[i, 1]
                dz = self.positions[j, 2] - self.positions[i, 2] 
                
                # Calculate Euclidean Distance
                dist_sq = dx**2 + dy**2 + dz**2 + self.softening**2
                dist_cubed = dist_sq ** 1.5

                # Force calculation
                force = self.G * self.masses[i, 0] * self.masses[j, 0] / dist_cubed

                forces[i, 0] += force * dx
                forces[i, 1] += force * dy
                forces[i, 2] += force * dz  # force on Z-axis
                
        return forces

    def compute_forces_numpy(self):
        """
        Vectorized O(N^2) using NumPy.
        """
        # Shape: (N, 1, 2) - (1, N, 2) -> (N, N, 2) 3D Tensor of all pairwise distances
        delta = self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :]

        # Euclidean distance squared (Shape: N, N)
        # axis=-1 tells NumPy to sum the squares across the last dimension
        dist_sq = np.sum(delta**2, axis=-1) + self.softening**2

        # Inverse distance cubed (Shape: N, N)
        inv_dist_cubed = dist_sq**-1.5

        # Final Force Vectors (Summed along columns to get total force per body)
        # Shape: (N, 2)
        force_magnitude = self.G * self.masses @ self.masses.T * inv_dist_cubed
        forces = np.sum(delta * force_magnitude[:, :, np.newaxis], axis=1)
        
        return forces

# ==========================================
# 🚀 THE 3D GRAVITY BENCHMARK
# ==========================================
num_planets = 1000
print(f"🌌 Booting True 3D N-Body Simulator with {num_planets} Planets...")
print(f"⚙️ Required calculations per frame: {num_planets * num_planets:,}\n")

sim = NBodySimulator3D(num_planets)

# 1. Benchmark Pure Python Loops
print("⏳ Running Pure Python Engine (3D Space)...")
start = time.time()
sim.compute_forces_python()
time_python = time.time() - start
print(f"🐍 Python Time : {time_python:.4f} seconds")

# 2. Benchmark NumPy Broadcasting
print("⏳ Running NumPy 3D Engine...")
start = time.time()
sim.compute_forces_numpy()
time_numpy = time.time() - start
print(f"⚡ NumPy Time   : {time_numpy:.4f} seconds")

# 3. Verdict
speedup = time_python / time_numpy
print("="*40)
print(f"🏆 NumPy is {speedup:.2f}x FASTER in 3D Space!")
print("="*40)