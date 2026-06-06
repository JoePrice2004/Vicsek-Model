import numpy as np
import matplotlib.pyplot as plt
from IPython import display

# Ensure proper rendering in Jupyter
  

# Parameters
N = 20  # Number of particles
L = 50.0  # Size of the domain
v0 = 1.0  # Desired speed of particles
r0 = 1.0  # Interaction radius
C = 150.0  # Force constant
dt = 3  # Time step
eta = 0.3  # Noise strength
steps = 2500  # Number of simulation steps
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

# Initialize the figure for three plots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))  # Three side-by-side plots

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
    ax2.clear()
    ax2.plot(order_params, label="Vicsek Order Parameter", color='r')
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Order Parameter (Φ)")
    ax2.set_title("Vicsek Model Order Parameter")
    ax2.legend()
    ax2.set_ylim(0, 1.1)  # Keep y-axis in range [0,1] for visibility

    # Compute and plot FFT over the full Vicsek order parameter history
    if len(order_params) > 10:  # Ensure we have enough data points
        full_data = np.array(order_params)  # Convert to NumPy array
        full_data -= np.mean(full_data)  # Remove DC component

        fft_result = np.fft.fft(full_data)  # Compute FFT
        frequencies = np.fft.fftfreq(len(full_data), dt)  # Compute frequency bins

        # Keep only positive frequencies
        positive_freqs = frequencies[:len(full_data) // 2]
        positive_fft = np.abs(fft_result[:len(full_data) // 2])

        # Normalize for better visibility
        max_x_scale = np.max(positive_fft) * 1.1 if np.max(positive_fft) > 0 else 1

        ax3.clear()
        ax3.plot(positive_freqs, positive_fft, color='g')  # Normal X-axis plot
        ax3.set_xlabel("Frequency")
        ax3.set_ylabel("Magnitude")
        ax3.set_title("Fourier Transform (Full Vicsek History)")
        ax3.set_xlim(0, np.max(positive_freqs))  # Set X-axis to max frequency range

    # Force Matplotlib to update display
    display.clear_output(wait=True)
    display.display(fig)
    plt.draw()
    plt.pause(0.001)  # Reduce pause time for smoother updates

plt.ioff()  # Disable interactive mode
plt.show()  # Show final state