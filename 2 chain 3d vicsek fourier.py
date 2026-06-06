import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython import display 

# Parameters
N = 100  # Number of particles (20 per chain)
L = 50.0  # Size of the domain
v0 = 1.0  # Desired speed of particles
r0 = 1.0  # Interaction radius
C = 150.0  # Force constant
dt = 0.1  # Time step
eta = 0.3  # Noise strength
steps = 2000  # Number of simulation steps
K = 100  # Strength of social forces

# Initialize positions and velocities
positions = np.random.rand(N, 3) * L
velocities = np.random.rand(N, 3) - 0.5
velocities = (velocities.T / np.linalg.norm(velocities, axis=1)).T * v0

# Compute nearest image vector for periodic boundary conditions
def compute_minimagevec(position1, position2, L):
    r12 = position2 - position1
    return np.where(np.abs(r12) < L / 2, r12, r12 - np.sign(r12) * L)

# Compute interaction forces (repel all opposite chain particles)
def compute_forces(positions):
    forces = np.zeros_like(positions)
    for i in range(N):
        for j in range(N):
            if i != j:
                rij = compute_minimagevec(positions[i], positions[j], L)
                dist = np.linalg.norm(rij)
                same_chain = (i < N // 2 and j < N // 2) or (i >= N // 2 and j >= N // 2)
                if dist <= r0:
                    # Always repel between chains
                    if not same_chain or abs(i - j) != 1:
                        forces[i] -= C * rij / (dist**5)
    return forces

# Compute social (leader-follower) forces within each chain
def compute_socialforces(positions):
    socialforces = np.zeros_like(positions)
    for i in range(N):
        if i % (N // 2) != 0:  # skip leaders (0 and 10)
            # Followers follow the previous particle in their own chain
            leader_index = i - 1
            rleader = compute_minimagevec(positions[i], positions[leader_index], L)
            dist = np.linalg.norm(rleader)
            socialforces[i] = K * rleader / dist
        # Add Langevin noise
        socialforces[i] += eta / np.sqrt(dt) * np.random.normal(0., 1., 3)
    return socialforces

# Update positions and velocities
def update(positions, velocities):
    forces = compute_forces(positions) + compute_socialforces(positions)
    velocities += forces * dt
    velocities = (velocities.T / np.linalg.norm(velocities, axis=1)).T * v0
    positions = (positions + velocities * dt) % L
    return positions, velocities

# Compute Vicsek order parameter
def compute_vicsek_order(velocities):
    avg_direction = np.sum(velocities / np.linalg.norm(velocities, axis=1)[:, None], axis=0) / N
    return np.linalg.norm(avg_direction)

# 7-point moving average
def smooth_data(data):
    smoothed_data = np.copy(data)
    for i in range(3, len(data) - 3):
        smoothed_data[i] = np.mean(data[i-3:i+4])
    return smoothed_data

# Setup plots
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)
order_params = []

for t in range(steps):
    for _ in range(4):
        positions, velocities = update(positions, velocities)

    # --- Plot Particle Positions ---
    ax1.clear()
    ax1.scatter(positions[:N//2, 0], positions[:N//2, 1], positions[:N//2, 2], c='b', label='Chain 1')
    ax1.scatter(positions[N//2:, 0], positions[N//2:, 1], positions[N//2:, 2], c='r', label='Chain 2')
    ax1.set_xlim(0, L)
    ax1.set_ylim(0, L)
    ax1.set_zlim(0, L)
    ax1.set_title(f"Step: {t+1}")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.legend()

    # --- Plot Vicsek Order Parameter ---
    order_params.append(compute_vicsek_order(velocities))
    smoothed_order = smooth_data(order_params) if len(order_params) > 6 else order_params

    ax2.clear()
    ax2.plot(smoothed_order, label="Smoothed Vicsek Order", color='r')
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Order Parameter (Φ)")
    ax2.set_title("Vicsek Order Parameter")
    ax2.legend()
    ax2.set_ylim(0, 1.1)

    # --- FFT Analysis ---


    plt.pause(0.001)

plt.show()
