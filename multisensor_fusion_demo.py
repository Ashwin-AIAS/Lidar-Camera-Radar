# multisensor_fusion_demo.py
# Simple multi-sensor fusion simulation: camera + LiDAR + radar
# - State: [x, y, vx, vy]
# - LiDAR: position (x,y) noisy, 10 Hz
# - Radar: velocity (vx, vy) noisy, 5 Hz (simulated as providing velocity vector)
# - Camera: position (x,y) noisy, 2 Hz and may miss detections (gives class "car" when detected)
# We fuse asynchronously with a linear Kalman Filter.

import numpy as np 
import matplotlib.pyplot as plt

np.random.seed(0)

# Simulation time
total_time = 20.0
dt = 0.02  # base time step (50 Hz)
t = np.arange(0, total_time + 1e-9, dt)
n = len(t)

# True trajectory: start at (0,0), initial velocity (1, 0.5), then a maneuver
true_x = np.zeros((n, 4))  # [x,y,vx,vy]
true_x[0] = np.array([0.0, 0.0, 1.0, 0.5])

for k in range(1, n):
    # simple maneuver: between 8s and 12s accelerate left/up
    a = np.array([0.0, 0.0])
    if 8.0 < t[k] < 12.0:
        a = np.array([-0.3, 0.2])
    true_x[k, 2:] = true_x[k-1, 2:] + a * dt
    true_x[k, :2] = true_x[k-1, :2] + true_x[k-1, 2:] * dt + 0.5 * a * dt**2

# Sensor schedules
lidar_rate = 10.0  # Hz
radar_rate = 5.0   # Hz
camera_rate = 2.0  # Hz

lidar_step = int(round(1.0/lidar_rate / dt))
radar_step = int(round(1.0/radar_rate / dt))
camera_step = int(round(1.0/camera_rate / dt))

lidar_indices = np.arange(0, n, lidar_step)
radar_indices = np.arange(0, n, radar_step)
camera_indices = np.arange(0, n, camera_step)

# Sensor noise parameters
lidar_pos_std = 0.2  # meters
radar_vel_std = 0.3  # m/s (radar measures velocity vector here for simplicity)
camera_pos_std = 1.5 # meters (camera less accurate in depth), and misses sometimes
camera_detection_prob = 0.85

# Generate sensor measurements (with some misses for camera)
lidar_meas = np.full((n,2), np.nan)
radar_meas = np.full((n,2), np.nan)
camera_meas = np.full((n,2), np.nan)
camera_detected = np.zeros(n, dtype=bool)

for idx in lidar_indices:
    pos = true_x[idx,:2] + np.random.randn(2) * lidar_pos_std
    lidar_meas[idx] = pos

for idx in radar_indices:
    vel = true_x[idx,2:] + np.random.randn(2) * radar_vel_std
    radar_meas[idx] = vel

for idx in camera_indices:
    if np.random.rand() < camera_detection_prob:
        pos = true_x[idx,:2] + np.random.randn(2) * camera_pos_std
        camera_meas[idx] = pos
        camera_detected[idx] = True
    else:
        # missed detection
        camera_detected[idx] = False

# Kalman Filter setup (discrete linear KF)
F = np.array([
    [1,0,dt,0],
    [0,1,0,dt],
    [0,0,1,0],
    [0,0,0,1]
])
B = np.zeros((4,2))  # no control input in this demo
# Process noise: assume acceleration noise
accel_noise_std = 0.5
q = accel_noise_std**2
Q = q * np.array([
    [0.25*dt**4, 0, 0.5*dt**3, 0],
    [0, 0.25*dt**4, 0, 0.5*dt**3],
    [0.5*dt**3, 0, dt**2, 0],
    [0, 0.5*dt**3, 0, dt**2]
])

# Measurement models (H) and covariances (R)
H_lidar = np.array([[1,0,0,0],[0,1,0,0]])  # measures position
R_lidar = (lidar_pos_std**2) * np.eye(2)

H_radar = np.array([[0,0,1,0],[0,0,0,1]])  # measures velocity vector (simplified)
R_radar = (radar_vel_std**2) * np.eye(2)

H_camera = H_lidar.copy()  # camera provides position (noisy)
R_camera = (camera_pos_std**2) * np.eye(2)

# Initialize filter
x_est = np.zeros((n,4))
P = np.eye(4) * 5.0  # large initial uncertainty
x = np.array([0.0, -0.5, 0.8, 0.3])  # initial guess slightly off

# For storing NIS for lidar and radar and camera updates
NIS_lidar = np.full(n, np.nan)
NIS_radar = np.full(n, np.nan)
NIS_camera = np.full(n, np.nan)

for k in range(n):
    # Prediction
    x = F.dot(x)
    P = F.dot(P).dot(F.T) + Q

    # LiDAR update (position)
    if not np.isnan(lidar_meas[k,0]):
        z = lidar_meas[k]
        z_pred = H_lidar.dot(x)
        y = z - z_pred
        S = H_lidar.dot(P).dot(H_lidar.T) + R_lidar
        K = P.dot(H_lidar.T).dot(np.linalg.inv(S))
        x = x + K.dot(y)
        P = (np.eye(4) - K.dot(H_lidar)).dot(P)
        NIS_lidar[k] = y.T.dot(np.linalg.inv(S)).dot(y)

    # Radar update (velocity)
    if not np.isnan(radar_meas[k,0]):
        z = radar_meas[k]
        z_pred = H_radar.dot(x)
        y = z - z_pred
        S = H_radar.dot(P).dot(H_radar.T) + R_radar
        K = P.dot(H_radar.T).dot(np.linalg.inv(S))
        x = x + K.dot(y)
        P = (np.eye(4) - K.dot(H_radar)).dot(P)
        NIS_radar[k] = y.T.dot(np.linalg.inv(S)).dot(y)

    # Camera update (position, occasionally missing)
    if camera_detected[k]:
        z = camera_meas[k]
        z_pred = H_camera.dot(x)
        y = z - z_pred
        S = H_camera.dot(P).dot(H_camera.T) + R_camera
        K = P.dot(H_camera.T).dot(np.linalg.inv(S))
        x = x + K.dot(y)
        P = (np.eye(4) - K.dot(H_camera)).dot(P)
        NIS_camera[k] = y.T.dot(np.linalg.inv(S)).dot(y)

    x_est[k] = x

# Compute errors
pos_error = np.linalg.norm(x_est[:,:2] - true_x[:,:2], axis=1)
vel_error = np.linalg.norm(x_est[:,2:] - true_x[:,2:], axis=1)

# Plots
plt.figure(figsize=(10,6))
plt.plot(true_x[:,0], true_x[:,1], label='True trajectory', linewidth=2)
plt.plot(x_est[:,0], x_est[:,1], label='KF estimate', linewidth=2)
plt.scatter(lidar_meas[lidar_indices,0], lidar_meas[lidar_indices,1], s=15, label='LiDAR meas', alpha=0.6)
plt.scatter(camera_meas[camera_indices,0], camera_meas[camera_indices,1], s=30, marker='x', label='Camera meas', alpha=0.6)
plt.quiver(true_x[radar_indices,0], true_x[radar_indices,1], true_x[radar_indices,2], true_x[radar_indices,3],
           color='grey', alpha=0.5, scale=10, label='True velocity (radar samples)')
plt.legend()
plt.title('Multi-sensor Fusion: True vs KF Estimate')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.grid(True)
plt.show()

plt.figure(figsize=(10,4))
plt.plot(t, pos_error)
plt.title('Position error (Euclidean norm)')
plt.xlabel('time [s]')
plt.grid(True)
plt.show()

plt.figure(figsize=(10,4))
plt.plot(t, vel_error)
plt.title('Velocity error (Euclidean norm)')
plt.xlabel('time [s]')
plt.grid(True)
plt.show()

# Print a short summary
print("Final position error: %.3f m" % pos_error[-1])
print("Final velocity error: %.3f m/s" % vel_error[-1])
from scipy.stats import chi2
m_pos = 2
lidar_nis_vals = NIS_lidar[~np.isnan(NIS_lidar)]
inside_lidar = np.sum((lidar_nis_vals >= chi2.ppf(0.025,m_pos)) & (lidar_nis_vals <= chi2.ppf(0.975,m_pos)))
frac_lidar = inside_lidar / len(lidar_nis_vals) if len(lidar_nis_vals)>0 else np.nan
print(f"LiDAR NIS inside 95%: {frac_lidar*100:.1f}% ({inside_lidar}/{len(lidar_nis_vals)})")
radar_nis_vals = NIS_radar[~np.isnan(NIS_radar)]
m_vel = 2
inside_radar = np.sum((radar_nis_vals >= chi2.ppf(0.025,m_vel)) & (radar_nis_vals <= chi2.ppf(0.975,m_vel)))
frac_radar = inside_radar / len(radar_nis_vals) if len(radar_nis_vals)>0 else np.nan
print(f"Radar NIS inside 95%: {frac_radar*100:.1f}% ({inside_radar}/{len(radar_nis_vals)})")
camera_nis_vals = NIS_camera[~np.isnan(NIS_camera)]
inside_camera = np.sum((camera_nis_vals >= chi2.ppf(0.025,m_pos)) & (camera_nis_vals <= chi2.ppf(0.975,m_pos)))
frac_camera = inside_camera / len(camera_nis_vals) if len(camera_nis_vals)>0 else np.nan
print(f"Camera NIS inside 95%: {frac_camera*100:.1f}% ({inside_camera}/{len(camera_nis_vals)})")
