import numpy as np
import matplotlib.pyplot as plt

# === Simulation Setup ===
dt = 0.05
N = 200

# Initial state: [x, y, theta]
x = np.array([1.7, 2.11, 0.0])
P = np.eye(3)

# Process and measurement noise
Q = 0.01 * np.eye(3)
R_imu = np.diag([1.0, 1.0, 0.01])  # ax, ay, yaw

# Measurement matrix (for yaw only)
H_imu = np.array([[0, 0, 1]])

# Simulated inputs
twist_inputs = np.array([[0.1, 0.0, 0.02]] * N)  # [vx, vy, omega]
imu_yaw_measurements = np.array([0.02] * N)

# Add outliers in the yaw measurements
imu_yaw_measurements[50] += 1.0   # Inject outlier
imu_yaw_measurements[120] += 2.5  # Another outlier

trajectory = []

# Predict function
def predict_state(x, u, dt):
    theta = x[2]
    dx = u[0] * np.cos(theta) - u[1] * np.sin(theta)
    dy = u[0] * np.sin(theta) + u[1] * np.cos(theta)
    dtheta = u[2]
    return x + dt * np.array([dx, dy, dtheta])

# Jacobian of the process model
def jacobian_F(x, u, dt):
    theta = x[2]
    return np.array([
        [1, 0, -dt * u[0] * np.sin(theta)],
        [0, 1,  dt * u[0] * np.cos(theta)],
        [0, 0, 1]
    ])

# Chi-square threshold for 1 DoF at 99% confidence
chi2_thresh = 6.63

# === Main EKF Loop ===
for t in range(N):
    u = twist_inputs[t]

    # Prediction step
    F = jacobian_F(x, u, dt)
    x = predict_state(x, u, dt)
    P = F @ P @ F.T + Q

    # Simulated IMU yaw measurement
    z = np.array([imu_yaw_measurements[t]])
    z_hat = H_imu @ x
    y = z - z_hat
    S = H_imu @ P @ H_imu.T + R_imu[2, 2]
    mahalanobis_distance = float(y.T @ np.linalg.inv(S) @ y)

    if mahalanobis_distance < chi2_thresh:
        # EKF Update
        K = P @ H_imu.T @ np.linalg.inv(S)
        x = x + (K.flatten() * y).flatten()
        P = (np.eye(3) - K @ H_imu) @ P
    else:
        print(f"[t={t}] IMU yaw measurement rejected: Mahalanobis^2 = {mahalanobis_distance:.2f}")

    trajectory.append(x[:2])

# === Plot Results ===
trajectory = np.array(trajectory)
plt.plot(trajectory[:, 0], trajectory[:, 1], label="EKF Trajectory", linewidth=2)
plt.xlabel("X position")
plt.ylabel("Y position")
plt.title("EKF with Mahalanobis Distance Outlier Rejection")
plt.grid(True)
plt.axis("equal")
plt.legend()
plt.show()
