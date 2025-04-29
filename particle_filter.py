import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# ========================
# STEP 1: Initialization
# ========================
print("STEP 1: Initializing parameters...")
N = 1000  # Number of particles
simTime = 100  # Reduced for testing
stateDim = 2
process_noise = 0.002
measurement_noise = 0.001

assert N > 0, "Particle count must be positive"
print(f"Initialized {N} particles for {simTime} timesteps")

# ========================
# STEP 2: Define Distributions
# ========================
print("\nSTEP 2: Creating noise distributions...")
process_dist = multivariate_normal(mean=[0, 0], cov=process_noise*np.eye(2))
measurement_dist = multivariate_normal(mean=[0], cov=measurement_noise)

samples = process_dist.rvs(size=5)
assert samples.shape == (5, 2), "Process noise shape mismatch"
print("Noise distributions validated")

# ========================
# STEP 3: System Dynamics
# ========================
print("\nSTEP 3: Setting up system matrices...")
m, ks, kd = 5, 200, 30
Ac = np.array([[0, 1], [-ks/m, -kd/m]])
Bc = np.array([[0], [1/m]])
Cc = np.array([[1, 0]])

h = 0.01  # Discretization step
A = np.eye(2) + h*Ac  # Euler discretization
B = h*Bc

print("A matrix:\n", A)
print("B matrix:\n", B)

# ========================
# STEP 4: True Trajectory
# ========================
print("\nSTEP 4: Generating ground truth...")
x_true = np.zeros((stateDim, simTime+1))
x_true[:, 0] = [0.1, 0.01]  # Initial state
control = 100 * np.ones((1, simTime))

for t in range(simTime):
    x_true[:, t+1] = A @ x_true[:, t] + B.flatten() * control[:, t] + process_dist.rvs()

assert not np.allclose(x_true, 0), "True trajectory generation failed"
print(f"True trajectory shape: {x_true.shape}")

# ========================
# STEP 5: Measurements
# ========================
print("\nSTEP 5: Simulating measurements...")
measurements = np.zeros(simTime)
for t in range(simTime):
    measurements[t] = Cc @ x_true[:, t] + measurement_dist.rvs()

print("Sample measurements:", measurements[:5])

# ========================
# STEP 6: Particle Initialization
# ========================
print("\nSTEP 6: Initializing particles...")
particles = np.random.uniform(low=[-0.5, -2], high=[1.5, 2], size=(N, 2)).T
weights = np.ones(N) / N

assert particles.shape == (2, N), "Particle matrix shape error"
print(f"Particles initialized with shape {particles.shape}")

# ========================
# STEP 7: Main Filter Loop
# ========================
print("\nSTEP 7: Running particle filter...")
est_states = np.zeros((stateDim, simTime))

for t in range(simTime):
    # ------------------
    # STEP 7.1: Prediction
    # ------------------
    particles = (A @ particles) + (B * control[:, t]) + process_dist.rvs(size=N).T
    assert not np.any(np.isnan(particles)), "NaN detected in prediction step"
    
    # ------------------
    # STEP 7.2: Update
    # ------------------
    innovations = measurements[t] - (Cc @ particles)
    weights = np.exp(-0.5 * innovations**2 / measurement_noise) * weights
    weights /= np.sum(weights)  # Normalize
    
    # ------------------
    # STEP 7.3: Resampling
    # ------------------
    Neff = 1 / np.sum(weights**2)
    if Neff < N/3:
        indices = np.random.choice(N, N, p=weights)
        particles = particles[:, indices]
        weights = np.ones(N) / N
    
    # ------------------
    # STEP 7.4: Estimation
    # ------------------
    est_states[:, t] = particles @ weights
    
    if t % 20 == 0:
        print(f"Timestep {t}: Neff = {Neff:.1f}, max weight = {np.max(weights):.4f}")

# ========================
# STEP 8: Validation
# ========================
print("\nSTEP 8: Validating results...")
errors = np.linalg.norm(est_states - x_true[:, :simTime], axis=0)
print(f"Mean error: {np.mean(errors):.4f}, Max error: {np.max(errors):.4f}")

# ========================
# STEP 9: Visualization
# ========================
print("\nSTEP 9: Plotting results...")
plt.figure(figsize=(12, 6))
plt.plot(x_true[0, :simTime], label='True Position')
plt.plot(est_states[0], '--', label='Estimated Position')
plt.xlabel('Time')
plt.ylabel('State')
plt.legend()
plt.title('Particle Filter Performance')
plt.grid(True)
plt.show()

# ========================
# STEP 10: Save Results
# ========================
print("\nSTEP 10: Saving outputs...")
np.savez('particle_filter_results.npz',
         true_states=x_true,
         est_states=est_states,
         measurements=measurements)
print("Results saved to particle_filter_results.npz")
