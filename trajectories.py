import numpy as np
rng = np.random.default_rng()

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

# TODO: make this more configurable later. B, Q, R should not be fixed, speed should not be 1deg/sec
def circular_traj_params(state_dim=2, input_dim=1, obs_dim=2):
    # Generate trajectories that follow a fixed path on the unit circle
    # returns: A, B, C, Q, R, x0, u_seq, traj, meas
    theta = 1/360*2*np.pi # one degree

    A = np.array([[np.cos(theta), -np.sin(theta)], # state transition matrix
                [np.sin(theta),  np.cos(theta)]]) # moving around a circle at 1 deg per timestep
    B = np.array([[0.5], [0.7]]) # input transformation
    # B = np.zeros(shape=(state_dim, input_dim)) # ignore inputs for now
    C = np.eye(obs_dim, state_dim) # Using identity map for now
    Q = 0.001*np.eye(state_dim) # Covariance matrix of process noise
    R = 0.01*np.eye(obs_dim) # Covariance matrix of sensor noise
    x0 = np.array([1.0, 0.0], dtype=np.float64) # starting state
    return A, B, C, Q, R, x0


def motion_traj_params(state_dim=2, input_dim=1, obs_dim=1):
    # Generate trajectories that traveling with a constant velocity
    # Only the position is observed.
    dt = 1e-3
    A = np.array([[1, dt], 
                  [0, 1]])
    B = np.array([[0], [1]])
    C = np.array([[1, 0]])
    Q = np.array([[0, 0], 
                [0, 1]])
    R = np.array([[0.4]])
    x0 = np.array([0.0, 0.0], dtype=np.float64) 
    return A, B, C, Q, R, x0



