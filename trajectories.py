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

# 2d rotation around a circle
def circular_traj_params():
    theta = 1/360*2*np.pi # one degree
    state_dim = 2
    input_dim = 1
    obs_dim = 2
    A = np.array([[np.cos(theta), -np.sin(theta)], # state transition matrix
                [np.sin(theta),  np.cos(theta)]]) # moving around a circle at 1 deg per timestep
    B = np.array([[0.5], [0.7]]) # input transformation
    C = np.eye(obs_dim, state_dim) # Using identity map for now
    Q = 0.001*np.eye(state_dim) # Covariance matrix of process noise
    R = 0.01*np.eye(obs_dim) # Covariance matrix of sensor noise
    x0 = np.array([1.0, 0.0], dtype=np.float64) # starting state
    return A, B, C, Q, R, x0, state_dim, input_dim, obs_dim

# Traveling with a constant velocity that can be driven, only the position is observed.
def motion_traj_params():
    state_dim = 2
    input_dim = 1
    obs_dim = 1
    dt = 1e-3
    A = np.array([[1, dt], 
                  [0, 1]])
    B = np.array([[0], [1]])
    C = np.array([[1, 0]]) # only observe the first hidden state
    Q = np.array([[0, 0], 
                [0, 1]])
    R = np.array([[0.4]])
    x0 = np.array([0.0, 0.0], dtype=np.float64) 
    return A, B, C, Q, R, x0, state_dim, input_dim, obs_dim

def so3_params():
    theta =  1/360*2*np.pi # one degree per timestep
    state_dim = 3
    input_dim = 1
    obs_dim = 3

    # Rotating along axis in middle of quadrant I
    A = Rotation.from_rotvec(np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]) * theta).as_matrix() 
    B = np.array([[0], [0], [1]])
    C = np.eye(state_dim)
    Q = 0.001*np.eye(state_dim)
    R = 0.01*np.eye(obs_dim)
    x0=np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return A, B, C, Q, R, x0, state_dim, input_dim, obs_dim

# Falling with constant acceleration. Velocity can be driven. Only position is observed.
# Process noise only affects the velocity. Sensor noise on the position.
def accel_traj_params():
    state_dim = 3
    input_dim = 1
    obs_dim = 1
    dt = 1e-3
    A = np.array([[1, dt, 0], 
                  [0, 1, dt], 
                  [0, 0, 1]])
    B = np.array([[0], [1], [0]])
    C = np.array([[1, 0, 0]])
    Q = 0.001 * np.eye(state_dim)
    R = 0.1 * np.eye(obs_dim)
    x0 = np.array([20, 0, -10]) # start x = 20, a = -10
    return A, B, C, Q, R, x0, state_dim, input_dim, obs_dim

def spring_mass_damper_traj_params():
    state_dim = 2
    input_dim = 1
    obs_dim = 1
    m = 1 # Mass
    k = 1 # Spring Constant
    b = 0.2 # Damping
    Ac = np.array([[ 0.0, 1.0], 
                   [-k/m, -b/m]])
    Bc = np.array([[0.0], [1/m]])
    Cc = np.array([[1.0, 0.0]])
    Q = 0.0001 * np.eye(state_dim)
    R = 0.01 * np.eye(obs_dim)
    x0 = np.array([1.0, 0.0])

    # model discretization
    sampling = 0.05 # sampling interval
    A = np.linalg.inv(np.eye(state_dim) - sampling*Ac)
    B = sampling * A @ Bc
    C = Cc

    return A, B, C, Q, R, x0, state_dim, input_dim, obs_dim

def identity_params():
    A = np.array([[1]])
    B = np.array([[0]])
    C = np.array([[1]])
    Q = np.array([[1]])
    R = np.array([[1]])
    x0 = np.array([1])
    state_dim = 1
    input_dim = 0
    obs_dim = 1
    return A, B, C, Q, R, x0, state_dim, input_dim, obs_dim

sys_params = {
    'identity': identity_params(),
    'circular': circular_traj_params(),
    'motion': motion_traj_params(),
    'so3': so3_params(),
    'accel': accel_traj_params(),
    'spring_mass_damper': spring_mass_damper_traj_params(),
}