import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
N = 20
L = 50.0
v0 = 1.0
r0 = 1.0
C = 150.0
dt = 0.1
eta = 0.3
steps = 3000
K = 100
save_steps = [1, 20, 50, 100, 150, 200]

# Initialize
positions = np.random.rand(N, 3) * L
velocities = np.random.rand(N, 3) - 0.5
velocities = (velocities.T / np.linalg.norm(velocities, axis=1)).T * v0

# Helper functions
def compute_minimagevec(position1, position2, L):
    r12 = position2 - position1
    return np.where(np.abs(r12) < L / 2, r12, r12 - np.sign(r12) * L)

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

def compute_socialforces(positions):
    socialforces = np.zeros_like(positions)
    for i in range(N):
        if i > 0:
            rleader = compute_minimagevec(positions[i], positions[i-1], L)
            dist = np.linalg.norm(rleader)
            socialforces[i] = K * rleader / dist
        socialforces[i] += eta / np.sqrt(dt) * np.random.normal(0., 1., 3)
    return socialforces

def update(positions, velocities):
    forces = compute_forces(positions) + compute_socialforces(positions)
    velocities += forces * dt
    velocities = (velocities.T / np.linalg.norm(velocities, axis=1)).T * v0
    positions = (positions + velocities * dt) % L
    return positions, velocities

def compute_vicsek_order(velocities):
    avg_direction = np.sum(velocities / np.linalg.norm(velocities, axis=1)[:, None], axis=0) / N
    return np.linalg.norm(avg_direction)

def smooth_data(data, window_size=50):
    kernel = np.ones(window_size) / window_size
    return np.convolve(data, kernel, mode='same')

# Setup for live simulation display
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122)
order_params = []

for t in range(steps):
    for _ in range(4):
        positions, velocities = update(positions, velocities)

    # Plot simulation
    ax1.clear()
    ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c='b')
    ax1.set_xlim(0, L)
    ax1.set_ylim(0, L)
    ax1.set_zlim(0, L)
    ax1.set_title(f"Step: {t+1}")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # Order parameter tracking
    order_params.append(compute_vicsek_order(velocities))
    if len(order_params) > 2:
        smoothed_order_params = smooth_data(order_params)
    else:
        smoothed_order_params = order_params

    ax2.clear()
    ax2.plot(smoothed_order_params, label="Smoothed Order Parameter", color='r')
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Order Parameter (Φ)")
    ax2.set_title("Vicsek Order Parameter")
    ax2.legend()
    ax2.set_ylim(0, 1.1)

    plt.pause(0.01)

    # Save snapshot at specified steps
    if (t + 1) in save_steps:
        plt.savefig(f"step_{t+1:03d}.png")

# After simulation ends, show Fourier Transform
plt.figure(figsize=(6, 4))
full_data = np.array(order_params)
full_data -= np.mean(full_data)
fft_result = np.fft.fft(full_data)
frequencies = np.fft.fftfreq(len(full_data), dt)
positive_freqs = frequencies[:len(full_data) // 2]
positive_fft = np.abs(fft_result[:len(full_data) // 2])

plt.plot(positive_freqs, positive_fft, color='g')
plt.xlabel("Frequency")
plt.ylabel("Magnitude")
plt.title("Fourier Transform (Full Order Parameter History)")
plt.xlim(0, np.max(positive_freqs))
plt.tight_layout()
plt.savefig("fft_output.png")
plt.show()
