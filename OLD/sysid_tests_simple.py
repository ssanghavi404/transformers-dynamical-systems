# System Identification in the General Case. Credit: Aleksander Haber, 
# https://github.com/AleksandarHaber/Subspace-Identification-State-Space-System-Identification-of-Dynamical-Systems-and-Time-Series-/blob/master/

import numpy as np
import matplotlib.pyplot as plt

###############################################################################
# This function estimates the Markov parameters of the state-space model:
# x_{k+1} =  A x_{k} + B u_{k} + Ke(k)
# y_{k}   =  C x_{k} + e(k)
# The function returns the matrix of the Markov parameters of the model
# Input parameters:

# "U" - is the input vector of the form U \in mathbb{R}^{m \times timeSteps}
# "Y" - is the output vector of the form Y \in mathbb{R}^{r \times timeSteps}
# "past" is the past horizon

# Output parameters:
#  The problem beeing solved is
#  min_{M_pm1} || Y_p_p_l -  M_pm1 Z_0_pm1_l ||_{F}^{2}
# " M_pm1" - matrix of the Markov parameters
# "Z_0_pm1_l" - data matrix used to estimate the Markov parameters,
# this is an input parameter for the "estimateModel()" function
# "Y_p_p_l" is the right-hand side 
def estimateMarkovParameters(U,Y,past):
    
    timeSteps=U.shape[1]
    m=U.shape[0]
    r=Y.shape[0]
    l=timeSteps-past-1
    
    # data matrices for estimating the Markov parameters
    Y_p_p_l=np.zeros(shape=(r,l+1))
    Z_0_pm1_l=np.zeros(shape=((m+r)*past,l+1))  # - returned
    # the estimated matrix that is returned as the output of the function
    M_pm1=np.zeros(shape=(r,(r+m)*past))   # -returned
    
    
    # form the matrices "Y_p_p_l" and "Z_0_pm1_l"
    # iterate through columns
    for j in range(l+1):
        # iterate through rows
        for i in range(past):
            Z_0_pm1_l[i*(m+r):i*(m+r)+m,j]=U[:,i+j]
            Z_0_pm1_l[i*(m+r)+m:i*(m+r)+m+r,j]=Y[:,i+j]
        Y_p_p_l[:,j]=Y[:,j+past]
        M_pm1=np.matmul(Y_p_p_l,np.linalg.pinv(Z_0_pm1_l))
    
    return M_pm1, Z_0_pm1_l, Y_p_p_l


###############################################################################
# This function estimates the state-space model:
# x_{k+1} =  A x_{k} + B u_{k} + Ke(k)
# y_{k}   =  C x_{k} + e(k)
# Acl= A - KC
    
# Input parameters:
    
# "U" - is the input matrix of the form U \in mathbb{R}^{m \times timeSteps}
# "Y" - is the output matrix of the form Y \in mathbb{R}^{r \times timeSteps}
# "Markov" - matrix of the Markov parameters returned by the function "estimateMarkovParameters()"
# "Z_0_pm1_l" - data matrix returned by the function "estimateMarkovParameters()"      
# "past" is the past horizon
# "future" is the future horizon
# Condition: "future" <= "past"
# "order_estimate" - state order estimate
    
# Output parameters:
# the matrices: A,Acl,B,K,C
# s_singular - singular values of the matrix used to estimate the state-sequence
# X_p_p_l   - estimated state sequence    
def estimateModel(U,Y,Markov,Z_0_pm1_l,past,future,order_estimate):
    
    timeSteps=U.shape[1]
    m=U.shape[0]
    r=Y.shape[0]
    l=timeSteps-past-1
    n=order_estimate
    
    Qpm1=np.zeros(shape=(future*r,past*(m+r)))
    for i in range(future):
        Qpm1[i*r:(i+1)*r,i*(m+r):]=Markov[:,:(m+r)*(past-i)]
    
    # estimate the state sequence
    Qpm1_times_Z_0_pm1_l=np.matmul(Qpm1,Z_0_pm1_l)
    Usvd, s_singular, Vsvd_transpose = np.linalg.svd(Qpm1_times_Z_0_pm1_l, full_matrices=True)
    # estimated state sequence
    X_p_p_l=np.matmul(np.diag(np.sqrt(s_singular[:n])),Vsvd_transpose[:n,:])    
    
    
    X_pp1_pp1_lm1=X_p_p_l[:,1:]
    X_p_p_lm1=X_p_p_l[:,:-1]
    
    # form the matrices Z_p_p_lm1 and Y_p_p_l
    Z_p_p_lm1=np.zeros(shape=(m+r,l))
    Z_p_p_lm1[0:m,0:l]=U[:,past:past+l]
    Z_p_p_lm1[m:m+r,0:l]=Y[:,past:past+l]
    
    Y_p_p_l=np.zeros(shape=(r,l+1))
    Y_p_p_l=Y[:,past:]
        
    S=np.concatenate((X_p_p_lm1,Z_p_p_lm1),axis=0)
    ABK=np.matmul(X_pp1_pp1_lm1,np.linalg.pinv(S))
    
    C=np.matmul(Y_p_p_l,np.linalg.pinv(X_p_p_l))
    Acl=ABK[0:n,0:n]
    B=ABK[0:n,n:n+m]  
    K=ABK[0:n,n+m:n+m+r]  
    A=Acl+np.matmul(K,C)    
    return A,Acl,B,K,C,s_singular,X_p_p_l

###############################################################################
# This function simulates an open loop state-space model:
# x_{k+1} = A x_{k} + B u_{k}
# y_{k}   = C x_{k}
# starting from an initial condition x_{0}

# Input parameters:
# A,B,C - system matrices 
# U - the input matrix, its dimensions are \in \mathbb{R}^{m \times simSteps},  where m is the input vector dimension
# Output parameters:
# Y - simulated output - dimensions \in \mathbb{R}^{r \times simSteps}, where r is the output vector dimension
# X - simulated state - dimensions  \in \mathbb{R}^{n \times simSteps}, where n is the state vector dimension
def systemSimulate(A,B,C,U,x0):
    simTime=U.shape[1]
    n=A.shape[0]
    r=C.shape[0]
    X=np.zeros(shape=(n,simTime+1))
    Y=np.zeros(shape=(r,simTime))
    for i in range(0,simTime):
        if i==0:
            X[:,[i]]=x0
            Y[:,[i]]=np.matmul(C,x0)
            X[:,[i+1]]=np.matmul(A,x0)+np.matmul(B,U[:,[i]])
        else:
            Y[:,[i]]=np.matmul(C,X[:,[i]])
            X[:,[i+1]]=np.matmul(A,X[:,[i]])+np.matmul(B,U[:,[i]])
    
    return Y,X

###############################################################################
# This function estimates an initial state x_{0} of the model
# x_{k+1} = A x_{k} + B u_{k}
# y_{k}   = C x_{k}
# using the input and output state sequences: {(y_{i}, u_{i})| i=0,1,2,\ldots, h}
# Input parameters:
# "A,B,C" - system matrices
# "U" - is the input matrix of the form U \in mathbb{R}^{m \times timeSteps}
# "Y" - is the output matrix of the form Y \in mathbb{R}^{r \times timeSteps}
# "h" - is the future horizon for the initial state estimation

# Output parameters:
# "x0_est"
def estimateInitial(A,B,C,U,Y,h):
    n=A.shape[0]
    r=C.shape[0]
    m=U.shape[0]
    
    # define the output and input time sequences for estimation
    Y_0_hm1=Y[:,0:h]
    Y_0_hm1=Y_0_hm1.flatten('F')
    Y_0_hm1=Y_0_hm1.reshape((h*r,1))
    
    U_0_hm1=U[:,0:h]
    U_0_hm1=U_0_hm1.flatten('F')
    U_0_hm1=U_0_hm1.reshape((h*m,1))
    
    
    O_hm1=np.zeros(shape=(h*r,n))
    I_hm1=np.zeros(shape=(h*r,h*m))
    

    for i in range(h):
        O_hm1[(i)*r:(i+1)*r,:]=np.matmul(C, np.linalg.matrix_power(A,i))
        if i>0:
            for j in range(i-1):
                I_hm1[i*r:(i+1)*r,j*m:(j+1)*m]=np.matmul(C,np.matmul(np.linalg.matrix_power(A,i-j-1),B))
    x0_est=np.matmul(np.linalg.pinv(O_hm1),Y_0_hm1-np.matmul(I_hm1,U_0_hm1))
    return x0_est

    
###############################################################################
#   This function computes the prediction performances of estimated models
#   Input parameters:
#   - "Ytrue" - true system output, dimensions: number of system outputs X time samples
#   - "Ypredicted" - output predicted by the model: number of system outputs X time samples
###############################################################################
def modelError(Ytrue,Ypredicted,r,m,n):
    r=Ytrue.shape[0]
    timeSteps=Ytrue.shape[1]
    total_parameters=n*(n+m+2*r)
    
    error_matrix=Ytrue-Ypredicted
    Ytrue=Ytrue.flatten('F')
    Ytrue=Ytrue.reshape((r*timeSteps,1))
    Ypredicted=Ypredicted.flatten('F')
    Ypredicted=Ypredicted.reshape((r*timeSteps,1))
    error=Ytrue-Ypredicted
    
    relative_error_percentage=(np.linalg.norm(error,2)/np.linalg.norm(Ytrue,2))*100
    
    vaf_error_percentage = (1 - ((1/timeSteps)*np.linalg.norm(error,2)**2)/((1/timeSteps)*np.linalg.norm(Ytrue,2)**2))*100
    vaf_error_percentage=np.maximum(vaf_error_percentage,0)
    cov_matrix=(1/(timeSteps))*np.matmul(error_matrix,error_matrix.T)
    Akaike_error=np.log(np.linalg.det(cov_matrix))+(2/timeSteps)*(total_parameters)
    
    return relative_error_percentage, vaf_error_percentage, Akaike_error

###############################################################################
# This function estimates an initial state x_{0} of the model
# x_{k+1} = \tilde{A} x_{k} + B u_{k} + K y_{k}
# y_{k}   = C x_{k}
# using the input and output state sequences: {(y_{i}, u_{i})| i=0,1,2,\ldots, h}
# Input parameters:
# "\tilde{A},B,C, K" - system matrices of the Kalman predictor state-space model
# "U" - is the input matrix of the form U \in mathbb{R}^{m \times timeSteps}
# "Y" - is the output matrix of the form Y \in mathbb{R}^{r \times timeSteps}
# "h" - is the future horizon for the initial state estimation

# Output parameters:
# "x0_est"
def estimateInitial_K(Atilde,B,C,K,U,Y,h):
    
    Btilde=np.block([B,K])
    Btilde=np.asmatrix(Btilde)
    
    n=Atilde.shape[0]
    r=C.shape[0]
    m=U.shape[0]
    m1=r+m
    
    # define the output and input time sequences for estimation
    Y_0_hm1=Y[:,0:h]
    Y_0_hm1=Y_0_hm1.flatten('F')
    Y_0_hm1=Y_0_hm1.reshape((h*r,1))
    
    U_0_hm1=U[:,0:h]
    U_0_hm1=U_0_hm1.flatten('F')
    U_0_hm1=U_0_hm1.reshape((h*m,1))
    
    Z_0_hm1=np.zeros(shape=(h*m1,1))
    for i in range(h):
        Z_0_hm1[i*m1:i*m1+m,:]=U_0_hm1[i*m:i*m+m,:]
        Z_0_hm1[i*m1+m:i*m1+m1,:]=Y_0_hm1[i*(r):(i+1)*r,:]
        
    
    O_hm1=np.zeros(shape=(h*r,n))
    I_hm1=np.zeros(shape=(h*r,h*m1))
    

    for i in range(h):
        O_hm1[(i)*r:(i+1)*r,:]=np.matmul(C, np.linalg.matrix_power(Atilde,i))
        if i>0:
            for j in range(i-1):
                I_hm1[i*r:(i+1)*r,j*m1:(j+1)*m1]=np.matmul(C,np.matmul(np.linalg.matrix_power(Atilde,i-j-1),Btilde))
  
    x0_est=np.matmul(np.linalg.pinv(O_hm1),Y_0_hm1-np.matmul(I_hm1,Z_0_hm1))
    return x0_est

###############################################################################
# This function performs an open-loop simulation of the state-space model:
# x_{k+1} = Atilde x_{k} + B u_{k} +K y_{k}
# y_{k}   = C x_{k}
# starting from an initial condition x_{0} and y_{0}
# Note: 

# Input parameters:
# Atilde,B,C,K - system matrices 
# U - the input matrix, its dimensions are \in \mathbb{R}^{m \times simSteps},  where m is the input vector dimension
# Output parameters:
# Y - simulated output - dimensions \in \mathbb{R}^{r \times simSteps}, where r is the output vector dimension
# X - simulated state - dimensions  \in \mathbb{R}^{n \times simSteps}, where n is the state vector dimension
def systemSimulate_Kopen(Atilde,B,C,K,U,x0,y0):
    simTime=U.shape[1]
    n=Atilde.shape[0]
    r=C.shape[0]
    X=np.zeros(shape=(n,simTime+1))
    Y=np.zeros(shape=(r,simTime))
    for i in range(0,simTime):
        if i==0:
            X[:,[i]]=x0
            Y[:,[i]]=y0
            X[:,[i+1]]=np.matmul(Atilde,x0)+np.matmul(B,U[:,[i]])+np.matmul(K,y0)
        else:
            Y[:,[i]]=np.matmul(C,X[:,[i]])
            X[:,[i+1]]=np.matmul(Atilde,X[:,[i]])+np.matmul(B,U[:,[i]])+np.matmul(K,Y[:,[i]])
    
    return Y,X

###############################################################################
# This function a closed-loop simulation of the state-space model:
# x_{k+1} = Atilde x_{k} + B u_{k} +K y_{k}
# y_{k}   = C x_{k}
# starting from an initial condition x_{0} and y_{0}
# Note: 

# Input parameters:
# Atilde,B,C,K - system matrices 
# U - the input matrix, its dimensions are \in \mathbb{R}^{m \times simSteps},  where m is the input vector dimension
# Ymeas - the measured output    
# Output parameters:
# Y - simulated output - dimensions \in \mathbb{R}^{r \times simSteps}, where r is the output vector dimension
# X - simulated state - dimensions  \in \mathbb{R}^{n \times simSteps}, where n is the state vector dimension
def systemSimulate_Kclosed(Atilde,B,C,K,U,Ymeas,x0):
    simTime=U.shape[1]
    n=Atilde.shape[0]
    r=C.shape[0]
    X=np.zeros(shape=(n,simTime+1))
    Y=np.zeros(shape=(r,simTime))
    for i in range(0,simTime):
        if i==0:
            X[:,[i]]=x0
            Y[:,[i]]=Ymeas[:,[i]]
            X[:,[i+1]]=np.matmul(Atilde,x0)+np.matmul(B,U[:,[i]])+np.matmul(K,Ymeas[:,[i]])
        else:
            Y[:,[i]]=np.matmul(C,X[:,[i]])
            X[:,[i+1]]=np.matmul(Atilde,X[:,[i]])+np.matmul(B,U[:,[i]])+np.matmul(K,Ymeas[:,[i]])
    
    return Y,X

# test function for fourth order system example. 
def main():
    # define the system parameters
    m1=20  ; m2=20   ; k1=1000  ; k2=2000 ; d1=1  ; d2=5; 


    # define the continuous-time system matrices
    Ac=np.matrix([[         0,             1,      0,      0],
                [-(k1+k2)/m1 ,  -(d1+d2)/m1 , k2/m1 , d2/m1 ], 
                [           0 ,            0 ,      0 ,     1], 
                [       k2/m2,          d2/m2, -k2/m2, -d2/m2]])
    Bc=np.matrix([[0],[0],[0],[1/m2]])
    Cc=np.matrix([[1, 0, 0, 0]])

    ###############################################################################
    # parameter definition

    r=1; m=1 # number of inputs and outputs
    # total number of time samples
    time=300
    # discretization constant
    sampling=0.05

    # model discretization
    I=np.identity(Ac.shape[0]) 
    A=np.linalg.inv(I-sampling*Ac)
    B=A*sampling*Bc
    C=Cc

    # define an input sequence and initial state for the identification
    input_ident=np.random.rand(1,time)
    x0_ident=np.random.rand(4,1)

    #define an input sequence and initial state for the validation
    input_val=input_ident
    # x0_val=x0_ident

    # simulate the discrete-time system to obtain the input-output data for identification and validation
    Y_ident, X_ident=systemSimulate(A,B,C,input_ident,x0_ident)
    # Y_val, X_val=systemSimulate(A,B,C,input_val,x0_val)

    #  end of parameter definition
    ###############################################################################

    ###############################################################################
    # model estimation and validation

    # estimate the Markov parameters
    past_value=10 # this is the past window - p 
    Markov, Z, Y_p_p_l = estimateMarkovParameters(input_ident ,Y_ident,past_value)

    # estimate the system matrices
    for model_order in range(6):
        print("Model Order", model_order)
        Aid,Atilde,Bid,Kid,Cid,s_singular,X_p_p_l = estimateModel(input_ident,Y_ident,Markov,Z,past_value,past_value,model_order)  

        # estimate the initial state of the validation data
        h=10 # window for estimating the initial state
        # x0est=estimateInitial(Aid,Bid,Cid,input_val,Y_val,h)

        # # simulate the open loop model 
        # # Y_val_prediction, X_val_prediction = systemSimulate(Aid,Bid,Cid,input_val,x0est)
        
        # What happens if we apply the learned system to the ORIGINAL (ident) data? 
        x0est = estimateInitial(Aid, Bid, Cid, input_ident, Y_ident, h)
        Y_ident_prediction, X_ident_prediction = systemSimulate(Aid,Bid,Cid,input_ident,x0est)

        # compute the errors
        relative_error_percentage, vaf_error_percentage, Akaike_error = modelError(Y_ident,Y_ident_prediction,r,m,30)
        print('Final model relative error %f and VAF value %f' %(relative_error_percentage, vaf_error_percentage))

        # plot the prediction and the real output 
        plt.plot(Y_ident[0,:100],'k',label='Real output')
        plt.plot(Y_ident_prediction[0,:100],'r',label='Prediction')
        plt.legend()
        plt.xlabel('Time steps')
        plt.ylabel('Predicted and real outputs')
        plt.show()

if __name__ == "__main__":
    main()