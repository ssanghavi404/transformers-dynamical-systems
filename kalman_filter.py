# Adapted from EE 126 Kalman Filter Lab
import copy
import numpy as np
import torch

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
                print("Converged in", t, "iterations") 
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

        
class LearnedKFilter:
    # Basically just a Linear RNN

    def __init__(self, state_dim, input_dim, obs_dim=None, x0=None): # obs_dim as a parameter
        if obs_dim is None: obs_dim = state_dim

        self.Aprime = torch.eye(state_dim, state_dim, requires_grad=True) # to be learned
        self.Bprime = torch.zeros(state_dim, input_dim, requires_grad=True) # to be learned
        self.Gprime = torch.zeros(state_dim, obs_dim, requires_grad=True) # to be learned
        self.Cprime = torch.eye(obs_dim, state_dim, requires_grad=True) # to be learned? Give it a try. #

        self.state_size = state_dim
        self.input_size = input_dim
        self.obs_size = obs_dim
        
        self.starting_state = torch.from_numpy(x0) if x0 is not None else torch.zeros(state_dim, requires_grad=True)

        self.loss_func = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam([self.Aprime, self.Bprime, self.Gprime], lr=1e-3)  # self.Cprime (decoder) is learnable? Not for now.
        self.losses = []

    def predict(self, curr_state, measurement, u=None):
        # print("Aprime is", self.Aprime)
        # print("Bprime is", self.Bprime)
        # print("Gprime is", self.Gprime)
        # print("curr_state is", curr_state)
        # print("measurement is", measurement)
        # print("u is", u)
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

    def fit(self, measurements, inputs=None, eps=4, maxIt=500):
        '''Learn the Kalman Filter's parameters (A', B', G') from a single sequence of measurements in inputs
            Given KF dynamics   
                xhat_tp1 = A (I - K C) xhat_t + B u_t + A K y_t
            Learns the Kalman update parameters
                A' = A (I - K C) and B' = B and G' = A K
            from given measurements and inputs'''
        T = measurements.shape[0]
        curr_loss = float('inf')
        i = 1
        while curr_loss > eps and i < maxIt:
            seq_loss = None
            curr_estimate = self.starting_state
            if i % 1000 == 0: print('Iteration', i, ": Loss", curr_loss)
            for t in range(T-1):
                next_estimate = self.predict(curr_estimate, torch.tensor(measurements[t], requires_grad=False), torch.tensor(inputs[t], requires_grad=False) if inputs is not None else None)
                
                next_obs_estimate = self.Cprime @ next_estimate
                target = torch.tensor(measurements[t+1], requires_grad=False)
                curr_loss = self.loss_func(next_obs_estimate, target)
                if seq_loss is None: seq_loss = curr_loss
                else: seq_loss += curr_loss
                curr_estimate = next_estimate

            self.optimizer.zero_grad()
            seq_loss.backward()
            self.optimizer.step()

            i += 1 
            curr_loss = seq_loss.item()
            self.losses.append(curr_loss)
           
    def simulate(self, measurements, inputs=None):
        T = measurements.shape[0]
        states = np.zeros(shape=(T, self.state_size))
        curr_state = self.starting_state
        for t in range(T):
            next_state = self.predict(curr_state, torch.tensor(measurements[t], requires_grad=False), torch.tensor(inputs[t], requires_grad=False) if inputs is not None else None)
            states[t] = next_state.detach().numpy()
            curr_state = next_state
        return states


# OLD 
# Helper function to perform system identification. Works for systems where the C is assumed to be the identity (outputs are noisy direct measurements of the input states).
def system_id(measurements, t, x0=0, inputs=None):
    '''system_id(measurement_data, t, starting_state, inputs_data) -> A, B
    Performs system identification using the first t timesteps of data.
    B will be None if inputs is None.'''
    state_dim = measurements[0].shape[0]
    input_dim = 0 if inputs is None else inputs.shape[1]
    M = np.zeros(shape=(t*state_dim, state_dim*(state_dim+input_dim)), dtype=np.float64)
    c = np.zeros(shape=(t*state_dim, 1))
    
    for i in range(t):
        for n in range(state_dim):
            row = i*state_dim + n
            c[row] = measurements[i, n] 
            M[row, n*state_dim:(n+1)*state_dim] = measurements[i-1] if i > 0 else x0
            if inputs is not None: 
                M[row, state_dim*state_dim + n*input_dim : state_dim*state_dim + (n+1)*input_dim] = inputs[i]

    AB_found = np.linalg.lstsq(M, c, rcond=None)[0].T[0]
    A_found = AB_found[:state_dim*state_dim].reshape((state_dim, state_dim))
    B_found = AB_found[state_dim*state_dim:].reshape((state_dim, input_dim)) if inputs is not None else None
    return A_found, B_found


##############################################################################################################################################################################################################################
from sysid_tests_simple import *

def sys_id(measurements, inputs, model_order=2, window=10):
    U = inputs.T
    Y = measurements.T
    past = window # How many of the past timesteps to use?
    future = window
    Markov, Z, Y_p_p_l = estimateMarkovParameters(U, Y, past)
    Aid, Atilde, Bid, Kid, Cid, s_singular, X_p_p_l = estimateModel(U,Y,Markov,Z,past,future,model_order)  
    return Aid, Bid, Cid, X_p_p_l

### THE ABOVE SYSID CODE IS NOT WORKING RIGHT NOW, try to debug it later
