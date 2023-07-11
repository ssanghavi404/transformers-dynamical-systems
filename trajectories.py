import numpy as np
import scipy as sp
from scipy.spatial.transform import Rotation

from plotting import *

rng = np.random.default_rng()

# Given the system parameters and timelength, generate num_traj trajectories
def generate_traj(num_traj, T, A, B, C, Q, R, x0, u_seq, state_dim=None, input_dim=None, obs_dim=None):
    if state_dim is None: state_dim = A.shape[0]
    if input_dim is None: input_dim = B.shape[1]
    if obs_dim is None: obs_dim = C.shape[0]

    traj = np.zeros(shape=(num_traj, T, state_dim), dtype=np.float64)
    meas = np.zeros(shape=(num_traj, T, obs_dim), dtype=np.float64)

    # Generate a trajectory 
    for traj_index in range(num_traj):
        x = x0
        for i in range(T):
            u_t = u_seq[traj_index, i]
            w_t = rng.multivariate_normal(mean=np.zeros(state_dim), cov=Q) # process noise
            x = A @ x + B @ u_t + w_t # inputs
            v_t = rng.multivariate_normal(mean=np.zeros(obs_dim), cov=R) # sensor noise
            y = C @ x + v_t
            traj[traj_index, i] = np.real(x)
            meas[traj_index, i] = np.real(y)
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
    B = np.array([[0.0], [1.0]]) # input transformation
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

    # Default rotation axis is middle of first quadrant
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
def accel_params(start_height=20, accel=-10, process_noise=0.001, sensor_noise=0.1):
    state_dim = 3
    input_dim = 1
    obs_dim = 1
    dt = 1e-3
    
    # State space is [[x], [xdot], [xdotdot]]
    A = np.array([[1, dt, 0], 
                  [0, 1, dt], 
                  [0, 0, 1]])
    B = np.array([[0], [1], [0]]) # only the velocity can be driven
    C = np.array([[1, 0, 0]]) # only the position can be observed
    Q = process_noise * np.eye(state_dim)
    R = sensor_noise * np.eye(obs_dim)
    x0= np.array([start_height, 0, accel]) 
    return A, B, C, Q, R, x0, state_dim, input_dim, obs_dim

# i-th order stable system
def stable_sys_params(order_n=3, input_dim=1, process_noise=0.0001, sensor_noise=0.001):
    # Generate a stable or marginally stable system of order n, with k-dimensional inputs
    diag = np.zeros(shape=(order_n, order_n))

    # All complex eigenvalues should come in complex conjugate pairs. This way, they 
    for i in range(order_n // 2):
        r = rng.random()*2 - 1 # Between -1 and 1
        theta = rng.random() * np.pi # random angle between 0 and 180deg
        # Complex eigenvalues in a conjugate pair
        block = r * np.array([[ np.cos(theta), -np.sin(theta) ],  # rotation matrix
                              [ np.sin(theta),  np.cos(theta) ]]) 
        diag[2*i:2*(i+1),  2*i:2*(i+1)] = block
    # if n is odd: need to have at least one real eigenvalue
    if order_n % 2 == 1: 
        diag[-1, -1] = rng.random()*2 - 1
    
    P = np.random.normal(size=(order_n, order_n))
    state_dim = order_n
    obs_dim = order_n

    A = P @ diag @ np.linalg.inv(P)
    # print("A is", A)
    U, S, Vt = np.linalg.svd(A)
    # print("Singular Values are", S) 
    # print("Condition Number is", S[0] / S[-1])
    evals, evec = np.linalg.eig(A)
    # print("Eigenvalues are", evals)
    B = np.zeros(shape=(order_n, input_dim)); B[-1, 0] = 1   # input transformation
    C = np.eye(obs_dim, state_dim) # State fully observed
    Q = process_noise * np.eye(state_dim) # Covariance matrix of process noise
    R = sensor_noise * np.eye(obs_dim) # Covariance matrix of sensor noise

    # x0: Allow a mixture of initializations here
    rand = rng.random()
    if rand < 0.2: x0 = np.zeros(order_n, dtype=np.float64); # Starting state at zero
    elif rand < 0.7: # starting state at steady state (start at 0, burn 100 samples)
        x0 = np.zeros(order_n, dtype=np.float64) # Starting state
        for i in range(100):
            w_t = rng.multivariate_normal(mean=np.zeros(state_dim), cov=Q) # process noise
            x0 = A @ x0 + w_t
    else: x0 = rng.multivariate_normal(mean=np.zeros(state_dim), cov=Q)
    # print("x0 is", x0)

    return A, B, C, Q, R, x0, state_dim, input_dim, obs_dim

def nontrivial_sys_params(order_n=3, input_dim=1, process_noise=0.0001, sensor_noise=0.001):
    if rng.random() < 0.2: P, S, Vt = np.linalg.svd(rng.random(size=(order_n, order_n))*2 - 1); # random unitary matrix
    else: P = np.random.normal(size=(order_n, order_n))
    diag = np.zeros(shape=(order_n, order_n))
    curr_eig_index = 0
    while curr_eig_index < order_n: 
        if curr_eig_index == order_n - 1: # If we only have one more eigenvalue to generate, it must be real.
            eig = (rng.random() * 2 - 1)
            diag[curr_eig_index, curr_eig_index] = eig # random number between -1 and 1
            curr_eig_index += 1
        else:
            if rng.random() < 0.3: # With probability 30%, draw a real eigenvalue.
                eig = rng.random() * 2 - 1 # random number between -1 and 1
                mult = order_n+1
                while mult + curr_eig_index > order_n: # rejection sample if we go over
                    mult = np.random.poisson(lam=1) + 1# Poisson random variable to determine the multiplicity of the eigenvalue
                
                print("Placing eigenvalue", eig, "with multiplicity", mult)
                for i in range(mult):
                    diag[curr_eig_index, curr_eig_index] = eig
                    if i > 0: diag[curr_eig_index-1, curr_eig_index] = 1 # add a "1" above the diagonal for nontrivial entries
                    curr_eig_index += 1
            else:
                r = rng.random()*2 - 1 # Between -1 and 1
                theta = rng.random() * np.pi # random angle between 0 and 180deg
                print('Placing eigenvalues', r * np.cos(theta) , " +/- " , r * np.sin(theta), "j")
                # Complex eigenvalues in a conjugate pair
                block = r * np.array([  [ np.cos(theta), -np.sin(theta) ],  # rotation matrix
                                        [ np.sin(theta),  np.cos(theta) ]   ]) 
                diag[curr_eig_index:(curr_eig_index+2),  curr_eig_index:(curr_eig_index+2)] = block
                curr_eig_index += 2

    print("Diag is \n", diag)
    
    A = P @ diag @ np.linalg.inv(P)
    state_dim = order_n
    obs_dim = order_n
    B = np.zeros(shape=(order_n, 1)); B[-1, 0] = 1   # input transformation
    C = np.eye(obs_dim, state_dim) # State fully observed
    Q = process_noise * np.eye(state_dim) # Covariance matrix of process noise
    R = sensor_noise * np.eye(obs_dim) # Covariance matrix of sensor noise

    x0 = np.zeros(order_n, dtype=np.float64) # Starting state
    return A, B, C, Q, R, x0, state_dim, input_dim, obs_dim
   
def test_higher_order_systems():
    A, B, C, Q, R, x0, state_dim, input_dim, obs_dim = stable_sys_params(order_n=100, sensor_noise=0.001)
    num_traj, T = 3, 100
    u_seq = 0.05 * (rng.random(size=(num_traj, T, input_dim))*2 - 1)
    traj, meas = generate_traj(num_traj, T, A, B, C, Q, R, x0, u_seq, state_dim, input_dim, obs_dim)
    # plot_poses2d({"Trajectory": traj[2, :, :2], "Measured":meas[2, :, :2]})

# def test_nontrivial_systems():
#     for _ in range(100):
#         A, B, C, Q, R, x0, state_dim, input_dim, obs_dim = nontrivial_sys_params(order_n=6, input_dim=1)

sys_params = {
    'so2': so2_params,
    'so3': so3_params,
    'smd': smd_params,
    'motion': motion_params,
    'accel': accel_params,
    'sys': stable_sys_params
}

test_higher_order_systems()