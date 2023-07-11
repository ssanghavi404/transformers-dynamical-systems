# Adapted from EE 126 Kalman Filter Lab
import copy
import numpy as np
import torch
from helpers import system_id

myDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class KFilter:
    def __init__(self, A, B, C, Q, R, state=None):
        self.A = A
        self.B = B
        self.Q = Q # covariance of w (process noise) 
        self.C = C
        self.R = R # covariance of v (sensor noise)

        self.state_size = A.shape[0] 
        self.input_size = B.shape[1] 
        self.obs_size = C.shape[0]
        
        if state is None: self.state = np.zeros(self.state_size)
        else: self.state = state

        self.prev_P = np.zeros((self.state_size, self.state_size))
        self.P = np.zeros((self.state_size, self.state_size)) # covariance of observation at time t+1 given time t
        self.steady_state = False

        self.K = None # Kalman Gain, recalculated in update() function
    
    def measure(self):
        return self.C @ self.state

    def predict(self, u=None):
        # u is the input at the timestep
        self.prev_P = copy.deepcopy(self.P)
        if u is None: self.state = self.A @ self.state
        else: self.state = self.A @ self.state + self.B @ u
        self.P = self.A @ self.prev_P @ self.A.T + self.Q
        
    def update(self, measurement, t=0):
        if not self.steady_state:
            self.K = self.P @ self.C.T @ np.linalg.inv(self.C @ self.P @ self.C.T + self.R)
            self.P = (np.eye(self.state_size) - self.K @ self.C) @ self.P
            if np.allclose(self.P, self.prev_P): 
                self.steady_state = True
                print("Kalman Filter converged in", t, "iterations") 
        innovation = measurement - self.C @ self.state
        self.state = self.state + self.K @ innovation

    def simulate(self, measurements, inputs=None):
        # 'measurements' is an T x obs_size array, where T is the number of timesteps
        # 'inputs' is a T x input_size array, or None if there are no inputs
        T = measurements.shape[0]
        states = np.zeros(shape=(T, self.state_size))
        for t in range(T):
            if inputs is not None: self.predict(inputs[t])
            else: self.predict()

            self.update(measurements[t], t)
            states[t] = self.state
        return states

    def run_till_ss(self):
        state_init = copy.deepcopy(self.state)
        i = 0
        while not self.steady_state:
            i += 1
            self.predict()
            self.update(np.zeros(self.obs_size,))

        self.state = state_init
        return i

        
class LearnedKF:
    # Learned Kalman Filter - Basically just a Linear RNN
    # Nonlinear Optimization since there is multiplicative effect of weights at each timestep

    def __init__(self, state_dim, input_dim, obs_dim=None, x0=None, lr=1e-2, optim='adam'): # obs_dim as a parameter
        if obs_dim is None: obs_dim = state_dim

        self.state_size = state_dim
        self.input_size = input_dim
        self.obs_size = obs_dim

        self.Aprime = torch.eye(self.state_size, self.state_size, requires_grad=True, device=myDevice) # to be learned
        self.Bprime = torch.zeros(self.state_size, self.input_size, requires_grad=True, device=myDevice) # to be learned
        self.Gprime = torch.zeros(self.state_size, self.obs_size, requires_grad=True, device=myDevice) # to be learned
        self.Cprime = torch.eye(self.obs_size, self.state_size, requires_grad=True, device=myDevice) # to be learned

        self.starting_state = (torch.from_numpy(x0) if x0 is not None else torch.zeros(self.state_size, requires_grad=True)).to(myDevice)

        self.loss_func = torch.nn.MSELoss()
        if optim == 'adam': self.optimizer = torch.optim.Adam([self.Aprime, self.Bprime, self.Cprime, self.Gprime], lr=lr) 
        else: self.optimizer = torch.optim.SGD([self.Aprime, self.Bprime, self.Cprime, self.Gprime], lr=lr)
        self.losses = []

    def predict(self, curr_state, measurement, u=None):
        if u is None: 
            nextState = \
                self.Aprime @ curr_state + \
                self.Gprime @ measurement
        else: 
            nextState = \
                self.Aprime @ curr_state + \
                self.Bprime @ u + \
                self.Gprime @ measurement
        return nextState

    def fit(self, meas, u_seq=None, maxIt=20000, eps=1e-6, delta=1e-8):
        '''Learn the Kalman Filter's parameters (A', B', G') from a single sequence of measurements in inputs
            Given KF dynamics   
                xhat_tp1 = A (I - K C) xhat_t + B u_t + A K y_t
            Learns the Kalman update parameters
                A' = A (I - K C) and B' = B and G' = A K
            from given measurements and inputs'''
        T = meas.shape[0]
        meas_torch = torch.tensor(meas, requires_grad=False, device=myDevice)
        if u_seq is None: u_seq = np.zeros(shape=(T, self.input_size))
        u_torch = torch.tensor(u_seq, requires_grad=False, device=myDevice)

        prev_avg_seq_loss = 0
        avg_seq_loss = float('inf')
        i = 1
        # need to compare to optimal kalman filter
        stopping_condition = False
        while not stopping_condition:
            prev_avg_seq_loss = avg_seq_loss
            seq_loss = None
            curr_estimate = self.starting_state

            for t in range(T-1):
                next_estimate = self.predict(curr_estimate, meas_torch[t], u_torch[t] if u_seq is not None else None)
                next_obs_estimate = self.Cprime @ next_estimate
                target = meas_torch[t+1]
                curr_loss = self.loss_func(next_obs_estimate, target)
                if seq_loss is None: seq_loss = curr_loss
                else: seq_loss += curr_loss
                curr_estimate = next_estimate

            # Values of the parameters before passing the 
            curr_Aprime = self.Aprime.detach().clone().cpu().numpy() 
            curr_Bprime = self.Bprime.detach().clone().cpu().numpy()
            curr_Gprime = self.Gprime.detach().clone().cpu().numpy()
            curr_Cprime = self.Cprime.detach().clone().cpu().numpy()

            self.optimizer.zero_grad()
            seq_loss.backward()
            self.optimizer.step()

            change_in_params = \
                np.linalg.norm(self.Aprime.detach().cpu().numpy() - curr_Aprime) + \
                np.linalg.norm(self.Bprime.detach().cpu().numpy() - curr_Bprime) + \
                np.linalg.norm(self.Gprime.detach().cpu().numpy() - curr_Gprime) + \
                np.linalg.norm(self.Cprime.detach().cpu().numpy() - curr_Cprime)

            i += 1
            avg_seq_loss = seq_loss.item() / T
            self.losses.append(avg_seq_loss)
            print("Iteration", i, "avg_seq_loss", avg_seq_loss)

            if avg_seq_loss < eps: 
                print("Stopping because avg_seq_loss < eps"); stopping_condition = True
            elif i > maxIt:
                print("Stopping due to maximum iterations hit"); stopping_condition = True
            elif abs(avg_seq_loss - prev_avg_seq_loss) < delta: 
                print("Stopping because loss is not decreasing"); stopping_condition = True
            elif change_in_params < delta:
                print("change in params is", change_in_params)
                print("Stopping because parameters are not moving"); stopping_condition = True

        print("LearnedKF converged in %d iterations" % i)
           
    def simulate(self, measurements, inputs=None):
        T = measurements.shape[0]
        states = np.zeros(shape=(T, self.state_size))
        curr_state = self.starting_state
        
        for t in range(T):
            next_state = self.predict(curr_state, torch.tensor(measurements[t], requires_grad=False, device=myDevice), 
                                      torch.tensor(inputs[t], requires_grad=False, device=myDevice) if inputs is not None else None)
            states[t] = next_state.detach().cpu().numpy()
            curr_state = next_state
        return states
    
class CheatingLKF(LearnedKF):
    # Cheating version of Learned Kalman Filter that uses information of the true A, B, C, Q, R for initialization

    def __init__(self, A, B, C, Q, R, x0=None, lr=1e-2, optim='adam', state_dim=None, input_dim=None, obs_dim=None): # obs_dim as a parameter
        super().__init__(state_dim or A.shape[0], input_dim or B.shape[1], obs_dim or C.shape[0])

        kf = KFilter(A, B, C, Q, R, state=x0)
        kf.run_till_ss()

        # Initialize these to what the True Kalman Filter converges to. 
        self.Aprime = torch.tensor(A @ (np.eye(self.state_size) - kf.K @ C), requires_grad=True, device=myDevice)
        self.Bprime = torch.tensor(B, requires_grad=True, device=myDevice) 
        self.Gprime = torch.tensor(A @ kf.K, requires_grad=True, device=myDevice) 
        self.Cprime = torch.tensor(C, requires_grad=True, device=myDevice) 

class InformedLKF(LearnedKF):
    # Learned Kalman filter with a better initialization
    # Performs system ID and then sets up A, B matrices. C is taken to be identity.
    # Requires knowledge of the sensor and process covariance matrices 

    def __init__(self, Q, R, meas, u_seq, x0=None, lr=1e-2, optim='adam'):
        super().__init__(state_dim=Q.shape[0], input_dim=u_seq.shape[1], obs_dim=R.shape[0], x0=x0, lr=lr, optim=optim)

        self.meas = meas
        self.u_seq = u_seq

        A_found, B_found = system_id(meas, len(u_seq), x0, u_seq)
        C = np.eye(shape=(self.state_dim, self.obs_dim))
        kf = KFilter(A_found, B_found, C, Q, R, state=x0)
        kf.run_till_ss()

        self.Aprime = torch.tensor(A_found @ (np.eye(self.state_dim) - kf.K @ C), requires_grad=True, device=myDevice)
        self.Bprime = torch.tensor(B_found, requires_grad=True, device=myDevice) 
        self.Gprime = torch.tensor(A_found @ kf.K, requires_grad=True, device=myDevice) 
        self.Cprime = torch.tensor(C, requires_grad=True, device=myDevice) 

    def fit(self, maxIt=20000, eps=1e-6, delta=1e-8):
        super().fit(self, self.meas, self.u_seq, maxIt, eps, delta)
        