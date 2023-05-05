
import numpy as np
import cvxpy as cp

# Helper function to perform system identification. Works for systems where the C is assumed to be the identity (outputs are noisy direct measurements of the input states).
def system_id(meas, t, x0=0, inputs=None):
    '''system_id(measurement_data, t, starting_state, inputs_data) -> A, B
    Performs system identification using the first t timesteps of data.
    B will be None if inputs is None.'''
    state_dim = meas[0].shape[0]
    input_dim = 0 if inputs is None else inputs.shape[1]
    M = np.zeros(shape=(t*state_dim, state_dim*(state_dim+input_dim)), dtype=np.float64)
    c = np.zeros(shape=(t*state_dim, 1))
    
    for i in range(t):
        for n in range(state_dim):
            row = i*state_dim + n
            c[row] = meas[i, n] 
            M[row, n*state_dim:(n+1)*state_dim] = meas[i-1] if i > 0 else x0
            if inputs is not None: 
                M[row, state_dim*state_dim + n*input_dim : state_dim*state_dim + (n+1)*input_dim] = inputs[i]
    AB_found = np.linalg.lstsq(M, c, rcond=None)[0].T[0]
    A_found = AB_found[:state_dim*state_dim].reshape((state_dim, state_dim))
    B_found = AB_found[state_dim*state_dim:].reshape((state_dim, input_dim)) if inputs is not None else None
    return A_found, B_found

# Helper function to optimize estimator. 
def optimal_traj(A, B, C, Q, R, meas, x0, u_seq):
    Qinv = np.linalg.inv(Q)
    Rinv = np.linalg.inv(R)
    state_dim = A.shape[0]
    T = meas.shape[0]

    # Least Squares Optimization with CVX: Minimum Energy Noise
    xs = cp.Variable((T, state_dim))

    # Set up the objective function
    obj = 0
    for i in range(1, T):
        w_hyp = xs[i, :].T - A @ xs[i-1, :].T - B @ u_seq[i, :].T
        obj += cp.quad_form(w_hyp, Qinv) # Minimize process noise
    for i in range(T):
        v_hyp = meas[i, :] - C @ xs[i, :].T
        obj += cp.quad_form(v_hyp, Rinv) # Minimize sensor noises

    # Special handling for the first state
    w_hyp0 = xs[0, :] - A @ x0 - B @ u_seq[0]     
    obj += cp.quad_form(w_hyp0, Qinv)

    # Setup and solve CVXPY problem with the objective above.
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve()
    ls_rec = xs.value

    return ls_rec
