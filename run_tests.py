## Helper Functions to run the tests
import numpy as np
import torch
torch.set_default_tensor_type(torch.DoubleTensor)
rng = np.random.default_rng()
from tqdm import tqdm
import matplotlib.pyplot as plt

myDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from kalman_filter import KFilter, LearnedKF, CheatLKF
from helpers import system_id, optimal_traj
from plotting import plot
import matplotlib.pyplot as plt
from trajectories import stable_sys_params, generate_traj


def recover(method: str, num_traj: int, T: int, 
            A: np.array, B: np.array, C: np.array, Q: np.array, R: np.array, x0: np.array, 
            traj: np.array, meas: np.array, u_seq: np.array, 
            state_dim: int, input_dim: int, obs_dim: int):
    '''
    Method: str
    Recover the filtered trajectory according to one of the following methods:
        'zero': Predict zero for every timestep.
        'prev': Just predict the previous token we saw
        'ls_opt': Optimization-based estimation of sequence of states that minimize energy of noise using CVX
        'kf': Kalman filter with time-varying gains, recalculated each timestep, using known A, B, C, Q, R
        'kf_ss': Kalman filter with pure steady-state gains, using known A, B, C, Q, R
        'kf_smooth': Forward-backward Kalman filter for smoothing, using known A, B, C, Q, R
        'id_kf': System Identification with Least Squares at each timestep + Kalman Filter update at each timestep (C is assumed to be identity)
        'id_kf_sim': System Identification with Least Squares at each timestep + Kalman Filter simulated from the start for each timestep
        'learn_kf': Learned Kalman Filter parameters using Pytorch (Basically a Linear RNN)  
        'cheat_lkf': Learned Kalman Filter parameters, initialized to true KF parameters
    Returns 
    '''
    recv = np.zeros(shape=(num_traj, T, state_dim))
    if method == 'zero':
        recv = np.zeros(shape=(num_traj, T, state_dim)) # just return the zeros
    elif method == 'prev': 
        recv[:, 1:, :] = meas[:, :-1, :]
        for i in range(num_traj): recv[i, 0] = x0
    elif method == 'ls_opt':
        for i in range(num_traj):
            recv[i] = optimal_traj(A, B, C, Q, R, meas[i], x0, u_seq[i])
    elif method == 'kf':
        for i in range(num_traj):
            kinematics_forward = KFilter(A, B, C, Q, R, state=x0)
            recv[i] = kinematics_forward.simulate(meas[i], u_seq[i])
    elif method == 'kf_ss':
        for i in range(num_traj):
            kinematics_forward = KFilter(A, B, C, Q, R, state=x0)
            kinematics_forward.run_till_ss()
            recv[i] = kinematics_forward.simulate(meas[i], u_seq[i])
    elif method == 'kf_smooth':
        for i in range(num_traj):
            kinematics_forward = KFilter(A, B, C, Q, R, state=x0) # Forward KF
            fltr_fwd = kinematics_forward.simulate(meas[i], u_seq[i])
            kinematics_backward = KFilter(np.linalg.inv(A), -np.linalg.inv(A) @ B, C, Q, R, state=traj[i][-1]) # Backward KF
            fltr_bkwd = np.flip(kinematics_backward.simulate(np.flip(meas[i], axis=0), np.flip(u_seq[i], axis=0)), axis=0)
            recv[i] = (fltr_fwd + fltr_bkwd) / 2
    elif method == 'id_kf':
        for i in range(num_traj):
            # Use the same data as before, but now no peeking on what are the actual A, B matrices
            A_unk, B_unk = np.zeros(shape=(state_dim, state_dim)), np.zeros(shape=(state_dim, input_dim))
            kinematics = KFilter(A_unk, B_unk, C, Q, R, state=x0)
            for t in range(T):
                A_found, B_found = system_id(meas[i], t, x0, u_seq[i])
                kinematics.A = A_found
                kinematics.B = B_found
                kinematics.predict(u_seq[i, t])
                kinematics.update(meas[i, t])
                recv[i, t] = kinematics.state
    elif method == 'id_kf_sim':
        for i in range(num_traj):
            for t in range(1, T):
                A_found, B_found = system_id(meas[i], t, x0, u_seq[i])
                kinematics = KFilter(A_found, B_found, C, Q, R, x0)
                recv[i, t] = kinematics.simulate(meas[i, :t], u_seq[i, :t])[-1]
    elif method == 'learn_kf':
        for i in range(num_traj):
            kinematics = LearnedKF(state_dim, input_dim, obs_dim, x0, lr=1e-2)
            kinematics.fit(meas[i], u_seq[i], eps=2e-3, maxIt=15000)
            learned_kf_sim = kinematics.simulate(meas[i], u_seq[i])
            recv[i] = learned_kf_sim
    elif method == 'cheat_lkf':
        for i in range(num_traj):
            kf = KFilter(A, B, C, Q, R, state=x0)
            kf.run_till_ss()
            kinematics = CheatLKF(A, B, C, kf.K, state_dim, input_dim, obs_dim, x0, lr=1e-2)
            kinematics.fit(meas[i], u_seq[i], eps=2e-3, maxIt=15000)
            learned_kf_sim = kinematics.simulate(meas[i], u_seq[i])
            recv[i] = learned_kf_sim
    else: return "Invalid method"
    return recv
    

def test_baselines(order_n: int, num_traj: int, T: int, methods, show_plot: bool=True):
    '''Test one of the baseline methods
    order_n: int, order of the system
    task_args: dict {kwarg:value} that are passed to generate trajectories
    num_traj: int, number of tests to perform
    T: int, how many timesteps to simulate the trajectory
    method: str in {'zero', 'prev', 'ls_opt', 'kf', 'kf_ss', 'kf_smooth', 'id_kf', 'id_kf_sim', 'learn_kf'}
    plot: bool, whether or not to plot the results (default True)

    Returns: traj, meas, recv
        traj: original trajectories shape=(num_traj, T, state_dim)
        meas: measurements shape=(num_traj, T, obs_dim)
        recv: recovered states shape=(num_traj, T, state_dim)
    '''
    A, B, C, Q, R, x0, state_dim, input_dim, obs_dim = stable_sys_params(order_n)
    u_seq = 0.05 * (rng.random(size=(num_traj, T, input_dim))*2 - 1)
    traj, meas = generate_traj(num_traj, T, A, B, C, Q, R, x0, u_seq, state_dim, input_dim, obs_dim)
    recovered = []
    for method in methods:
        recv = recover(method, num_traj, T, A, B, C, Q, R, x0, traj, meas, u_seq, state_dim, input_dim, obs_dim)
        recovered.append(recv)
    if show_plot: plot({"Trajectory":traj[0], "Measured":meas[0], "Recovered":recv[0]})

    return traj, meas, recovered

def run_tests(num_tests=100, traj_len=250):  
    for order_n in range(3, 4): #, 6): # Order 3, 4, 5,  
        # print("order_n", order_n)
        plt.figure()
        plt.yscale('log')
        colors = ['k', 'r', 'g', 'b', 'y']
        methods = ['zero', 'prev', 'id_kf', 'learn_kf', 'cheat_lkf'] 
        traj, meas, recovered = test_baselines(order_n, num_tests, traj_len, methods, show_plot=False)
        for methodNum in range(len(methods)):
            method = methods[methodNum]
            recv = recovered[methodNum]
            num_tests, traj_len, state_dim = traj.shape
            errs = np.zeros(shape=(num_tests, traj_len))
            for testNum in range(num_tests):
                for t in range(traj_len):
                    errs[testNum, t] = np.linalg.norm(traj[testNum, t] - recv[testNum, t])**2
            q1s, meds, q3s = [], [], []
            for t in range(5, traj_len):
                q1s.append(np.quantile(errs[:, t], 0.25))
                meds.append(np.median(errs[:, t]))
                q3s.append(np.quantile(errs[:, t], 0.75))
            plt.scatter(range(len(q1s)), q1s, color=colors[methodNum], label=method)
            plt.scatter(range(len(meds)), meds, color=colors[methodNum])
            plt.scatter(range(len(q3s)), q3s, color=colors[methodNum])
        plt.title("System of order {0}".format(order_n))    
        plt.xlabel("Timestep")
        plt.ylabel("Error at Timestep")
        plt.legend()
        plt.show()

run_tests()