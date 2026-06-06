import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython import display 

# Parameters
N = 20  # Number of particles
L = 50.0  # Size of the domain
v0 = 1.0  # Desired speed of particles
r0 = 1.0  # Interaction radius
C = 150.0  # Force constant
dt = 0.1   # Time step
eta = 0.3  # Noise strength
steps = 2000  # Number of simulation steps
K = 100  # Strength of social forces

# Initialize positions and velocities
positions = np.random.rand(N, 3) * L
velocities = np.random.rand(N, 3) - 0.5
velocities = (velocities.T / np.linalg.norm(velocities, axis=1)).T * v0

# Periodic boundary condition helper
def compute_minimagevec(position1, position2, L):
    r12 = position2 - position1
    return np.where(np.abs(r12) < L / 2, r12, r12 - np.sign(r12) * L)

# Repulsion forces
def compute_forces(positions):
    forces = np.zeros_like(positions)
    for i in range(N):
        for j in range(N):
            if i != j:
                rij = compute_minimagevec(positions[i], positions[j], L)
                dist = np.linalg.norm(rij)
                if dist <= r0:
                    forces[i] -= C * rij / (dist**5)
    return forces

# Social forces with noise and leader-following
def compute_socialforces(positions):
    socialforces = np.zeros_like(positions)
    for i in range(N):
        if i > 0:
            rleader = compute_minimagevec(positions[i], positions[i-1], L)
            dist = np.linalg.norm(rleader)
            socialforces[i] = K * rleader / dist
        socialforces[i] += eta / np.sqrt(dt) * np.random.normal(0., 1., 3)
    return socialforces

# Update rule
def update(positions, velocities):
    forces = compute_forces(positions) + compute_socialforces(positions)
    velocities += forces * dt
    velocities = (velocities.T / np.linalg.norm(velocities, axis=1)).T * v0
    positions = (positions + velocities * dt) % L
    return positions, velocities

# Vicsek order parameter
def compute_vicsek_order(velocities):
    avg_direction = np.sum(velocities / np.linalg.norm(velocities, axis=1)[:, None], axis=0) / N
    return np.linalg.norm(avg_direction)

# Moving average smoothing
def smooth_data(data, window_size=50):
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='same')

# Set up plotting
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(131, projection='3d')
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

order_params = []

# --- Simulation Loop ---
for t in range(steps):
    for _ in range(4):
        positions, velocities = update(positions, velocities)

    # Update 3D particle plot
    ax1.clear()
    ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='b')
    ax1.set_xlim(0, L)
    ax1.set_ylim(0, L)
    ax1.set_zlim(0, L)
    ax1.set_aspect('equal', 'box')
    ax1.set_title(f"Step: {t+1}")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # Update order parameter
    order_params.append(compute_vicsek_order(velocities))
    if len(order_params) >= 50:
        smoothed_order_params = smooth_data(order_params, window_size=50)
    else:
        smoothed_order_params = order_params

    # Plot order parameter
    ax2.clear()
    ax2.plot(smoothed_order_params, label="Smoothed Order Parameter", color='r')
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Order Parameter (Φ)")
    ax2.set_title("Vicsek Order Parameter")
    ax2.set_ylim(0, 1.1)
    ax2.legend()

    plt.pause(0.001)

# --- FFT after simulation ends ---
fig_fft = plt.figure(figsize=(6, 4))
ax_fft = fig_fft.add_subplot(111)

full_data = np.array(order_params)
full_data -= np.mean(full_data)  # Remove DC offset

fft_result = np.fft.fft(full_data)
frequencies = np.fft.fftfreq(len(full_data), dt)

# Positive frequencies only
positive_freqs = frequencies[:len(full_data) // 2]
positive_fft = np.abs(fft_result[:len(full_data) // 2])

ax_fft.plot(positive_freqs, positive_fft, color='g')
ax_fft.set_xlabel("Frequency")
ax_fft.set_ylabel("Magnitude")
ax_fft.set_title("Fourier Transform (Full Order Parameter History)")
ax_fft.set_xlim(0, np.max(positive_freqs))

plt.show()
