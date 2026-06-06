import numpy as np
import matplotlib.pyplot as plt
from IPython import display

# Parameters
N = 20  # Number of particles
L = 50.0  # Size of the domain
v0 = 1.0  # Desired speed of particles
r0 = 1.0  # Interaction radius
C = 150.0  # Force constant
dt = 0.05  # Reduced time step for better frequency resolution
eta = 2.0  # Noise strength
steps = 5000  # Increased simulation steps for clearer periodicity
K = 100  # Strength of social forces

# Initialize positions and velocities
positions = np.random.rand(N, 2) * L
velocities = np.random.rand(N, 2) - 0.5
velocities = (velocities.T / np.linalg.norm(velocities, axis=1)).T * v0

# Compute nearest image vector for periodic boundary conditions
def compute_minimagevec(position1, position2, L):
    r12 = position2 - position1
    return np.where(np.abs(r12) < L / 2, r12, r12 - np.sign(r12) * L)

# Compute interaction forces to prevent overlaps
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

# Compute 'social' forces with leader-follower structure
def compute_socialforces(positions):
    socialforces = np.zeros_like(positions)
    for i in range(N):
        if i > 0:
            rleader = compute_minimagevec(positions[i], positions[i-1], L)
            dist = np.linalg.norm(rleader)
            socialforces[i] = K * rleader / dist
        socialforces[i] += eta / np.sqrt(dt) * np.random.normal(0., 1., 2)  # Langevin noise
    return socialforces

# Update positions and velocities
def update(positions, velocities):
    forces = compute_forces(positions) + compute_socialforces(positions)
    velocities += forces * dt
    velocities = (velocities.T / np.linalg.norm(velocities, axis=1)).T * v0
    positions = (positions + velocities * dt) % L  # Apply periodic boundary conditions
    return positions, velocities

# Compute Vicsek order parameter
def compute_vicsek_order(velocities):
    avg_direction = np.sum(velocities / np.linalg.norm(velocities, axis=1)[:, None], axis=0) / N
    return np.linalg.norm(avg_direction)

# Apply 20-point smoothing using a simple moving average for less smoothing
def smooth_data(data, window_size=20):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Initialize the figure for four plots
fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 6))  # Four side-by-side plots

plt.ion()  # Enable interactive mode

order_params = []  # Store order parameter for plotting

for t in range(steps):
    for _ in range(4):  # Internal updates per step
        positions, velocities = update(positions, velocities)

    # Clear and redraw particle simulation
    ax1.clear()
    ax1.scatter(positions[:, 0], positions[:, 1], c='b')
    ax1.set_xlim(0, L)
    ax1.set_ylim(0, L)
    ax1.set_aspect('equal', 'box')
    ax1.set_title(f"Step: {t+1}")

    # Compute and plot Vicsek order parameter
    order_params.append(compute_vicsek_order(velocities))
    smoothed_order_params = smooth_data(order_params) if len(order_params) >= 20 else order_params

    ax2.clear()
    ax2.plot(smoothed_order_params, label="Smoothed Order Parameter", color='r')
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Order Parameter (Φ)")
    ax2.set_title("Vicsek Model Order Parameter")
    ax2.legend()
    ax2.set_ylim(0, 1.1)

    # Compute and plot FFT over the smoothed Vicsek order parameter history
    if len(smoothed_order_params) > 10:
        full_data = np.array(smoothed_order_params)
        full_data -= np.mean(full_data)

        # Apply a Hanning window to reduce spectral leakage
        windowed_data = full_data * np.hanning(len(full_data))
        fft_result = np.fft.fft(windowed_data)
        frequencies = np.fft.fftfreq(len(full_data), dt)

        # Keep only positive frequencies
        positive_freqs = frequencies[:len(full_data) // 2]
        positive_fft = np.abs(fft_result[:len(full_data) // 2])

        ax3.clear()
        ax3.plot(positive_freqs, positive_fft, color='g')
        ax3.set_xlabel("Frequency")
        ax3.set_ylabel("Magnitude")
        ax3.set_title("Fourier Transform (Smoothed Vicsek History)")
        ax3.set_xlim(0, 5)

        # Compute and plot the inverse Fourier Transform
        inverse_fft = np.fft.ifft(fft_result).real
        ax4.clear()
        ax4.plot(range(len(inverse_fft)), inverse_fft, color='purple')
        ax4.set_xlabel("Time Step")
        ax4.set_ylabel("Inverse FFT")
        ax4.set_title("Inverse Fourier Transform")

    display.clear_output(wait=True)
    display.display(fig)
    plt.draw()
    plt.pause(0.001)

plt.ioff()
plt.show()
