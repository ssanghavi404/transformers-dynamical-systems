import numpy as np
import scipy as sp
from scipy.spatial.transform import Rotation

rng = np.random.default_rng()

# Given the system parameters and timelength, generate num_traj trajectories
def generate_traj(num_traj, T, A, B, C, Q, R, x0, u_seq, state_dim=None, input_dim=None, obs_dim=None):
    if state_dim is None: state_dim = A.shape[0]
    if input_dim is None: input_dim = B.shape[1]
    if obs_dim is None: obs_dim = C.shape[0]

    traj = np.zeros(shape=(num_traj, T, state_dim))
    meas = np.zeros(shape=(num_traj, T, obs_dim))

    # Generate a trajectory 
    for traj_index in range(num_traj):
        x = x0
        for i in range(T):
            u_t = u_seq[traj_index, i]
            w_t = rng.multivariate_normal(mean=np.zeros(state_dim), cov=Q) # process noise
            x = A @ x + w_t + B @ u_t # inputs
            v_t = rng.multivariate_normal(mean=np.zeros(obs_dim), cov=R) # sensor noise
            y = C @ x + v_t
            traj[traj_index, i] = x
            meas[traj_index, i] = y
    return traj, meas

# 2d rotation around a circle, rotate "angle" degrees in each timestep
def so2_params(angle=1, process_noise=0.001, sensor_noise=0.01):
    # Rotate "angle" degrees in each timestep
    theta = angle * 1/360*2*np.pi # one degree per timestep
    state_dim = 2
    input_dim = 1
    obs_dim = 2
    A = np.array([[np.cos(theta), -np.sin(theta)], # state transition matrix
                [np.sin(theta),  np.cos(theta)]]) # moving around a circle at 1 deg per timestep
    B = np.array([[0], [1]]) # input transformation
    C = np.eye(obs_dim, state_dim) # State fully observed
    Q = process_noise * np.eye(state_dim) # Covariance matrix of process noise
    R = sensor_noise * np.eye(obs_dim) # Covariance matrix of sensor noise
    x0= np.array([1.0, 0.0], dtype=np.float64) # Starting state
    return A, B, C, Q, R, x0, state_dim, input_dim, obs_dim


def so3_params(angle=1, axis=np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]), process_noise=0.001, sensor_noise=0.01):
    theta = angle * 1/360*2*np.pi # one degree per timestep
    state_dim = 3
    input_dim = 1
    obs_dim = 3

    # Defaulr rotation axis is middle of first quadrant
    A = Rotation.from_rotvec(axis * theta).as_matrix() 
    B = np.array([[0], [0], [1]])
    C = np.eye(state_dim)
    Q = process_noise * np.eye(state_dim)
    R = sensor_noise * np.eye(obs_dim)
    x0= np.array([1.0, 0.0, 0.0], dtype=np.float64) # starting state
    return A, B, C, Q, R, x0, state_dim, input_dim, obs_dim

def smd_params(mass=1, k_spring=1, b_damper=0.2, process_noise=0.0001, sensor_noise=0.01):
    state_dim = 2
    input_dim = 1
    obs_dim = 1
    m = mass # Mass
    k = k_spring # Spring Constant
    b = b_damper # Damping

    # State space is [[x], [xdot]]
    Ac = np.array([[ 0.0, 1.0], 
                   [-k/m, -b/m]]) # Continuous time dynamics
    Bc = np.array([[0.0], [1/m]]) # Continuous time input transformation

    # model discretization
    sampling = 0.05 # sampling interval
    A = np.linalg.inv(np.eye(state_dim) - sampling*Ac)
    B = sampling * A @ Bc

    C = np.array([[1.0, 0.0]]) # Only position x observed at each timestep
    Q = process_noise * np.eye(state_dim)
    R = sensor_noise * np.eye(obs_dim)
    x0= np.array([1.0, 0.0]) # Starting state

    return A, B, C, Q, R, x0, state_dim, input_dim, obs_dim


# Traveling with a constant velocity that can be driven, only the position is observed.
def motion_params(process_noise=1, sensor_noise=0.4):
    state_dim = 2
    input_dim = 1
    obs_dim = 1
    dt = 1e-3

    # State space is [[x], [xdot]]
    A = np.array([[1, dt], 
                  [0, 1]])
    B = np.array([[0], [1]])
    C = np.array([[1, 0]]) # only observe the position x
    Q = process_noise * np.array([[0, 0], 
                                  [0, 1]])
    R = sensor_noise * np.array([[1]])
    x0= np.array([0.0, 0.0], dtype=np.float64) 
    return A, B, C, Q, R, x0, state_dim, input_dim, obs_dim

# Falling with constant acceleration. Velocity can be driven. Only position is observed.
# Process noise only affects the velocity. Sensor noise on the position.
def accel_params():
    state_dim = 3
    input_dim = 1
    obs_dim = 1
    dt = 1e-3
    
    # State space is [[x], [xdot], [xdoubledot]]
    A = np.array([[1, dt, 0], 
                  [0, 1, dt], 
                  [0, 0, 1]])
    B = np.array([[0], [1], [0]]) # only the velocity can be driven
    C = np.array([[1, 0, 0]]) # only the position can be observed
    Q = 0.001 * np.eye(state_dim)
    R = 0.1 * np.eye(obs_dim)
    x0= np.array([20, 0, -10]) # start at height x = 20, acceleration a = -10
    return A, B, C, Q, R, x0, state_dim, input_dim, obs_dim

default_sys_params = {
    'so2': so2_params(),
    'so3': so3_params(),
    'smd': smd_params(),
    'motion': motion_params(),
    'accel': accel_params(),
}