import scipy
import numpy as np 

def cox_de_boor(x, k, i, t):
    """Cox-de Boor recursion formula to calculate the value of the ith basis function of degree k at x."""
    if k == 0:
        return 1.0 if t[i] <= x < t[i+1] else 0.0
    else:
        term1 = 0.0 if t[i+k] == t[i] else (x - t[i]) / (t[i+k] - t[i]) * cox_de_boor(x, k-1, i, t)
        term2 = 0.0 if t[i+k+1] == t[i+1] else (t[i+k+1] - x) / (t[i+k+1] - t[i+1]) * cox_de_boor(x, k-1, i+1, t)
        return term1 + term2

def generate_basis_functions(degree, knot_vector, num_points=100):
    """Generate B-spline basis functions of a given degree with a specified knot vector."""
    x_values = np.linspace(knot_vector[0], knot_vector[-1], num_points)
    basis_functions = np.zeros((len(knot_vector)-1-degree, num_points))
    
    for i in range(len(knot_vector)-1-degree):
        for j, x in enumerate(x_values):
            basis_functions[i, j] = cox_de_boor(x, degree, i, knot_vector)
    
    return x_values, basis_functions

def build_toeplix_matrix(r, total_lag):
    # build the Toepliz matrix from the sparse event data in r
    Alist = []
    for i in range(r.shape[1]):
        A = scipy.linalg.convolution_matrix(r[:,i], total_lag, 'same')
        Alist.append(A)
    return np.hstack(Alist)


def forward_model(x, A, P, kernel_size=40):
    '''
    A : Toeplitz matrix of sparse event
    P: predictor
    '''
    kernels = x[:A.shape[1]].reshape(-1,1)
    beta = x[A.shape[1]:].reshape(-1,2)

    x = np.sum(A*(kernels*(beta@P)).T,axis=1)
    # print(x.shape)
    # print((A@kernels).shape)
    
    return np.sum(A*(beta@P).T, axis=1, keepdims=True) + A@kernels


def prepare_pred_matrix(P, r):
    '''
    P: (num_beta x time) matrix of predictor
    r: (num_event x time) sparse matrix indicating events

    returns:
    P_flat: (num_beta*num_event x time) flat matrix of the predictor multiplied with the sparse event matrix
    '''
    num_events = r.shape[0]
    # duplicate the predictor for all the event kernels
    Pm = np.tile(P, (num_events,1,1)) #predictor matrix
    # print(Pm.shape)

    # multiple the predictor with the event to prepare for convolution

    for i in range(num_events):
        # print(Pm[i,:,:].shape,r.T[i,:].shape)
        Pm[i,:,:] *= r[i,:]
    
    P_flat= Pm.reshape(-1, Pm.shape[2]) # flatten to 2D for easier convolution

    return P_flat

def collect_events(evt, data, kernel_size=41):
    # collect the event waveform around data
    # evt should be a binary matrix with 1 indicating the occurance of an event

    assert len(evt)==len(data)
    
    evt_idx = np.nonzero(evt)[0]
    evt_idx = evt_idx[(evt_idx>kernel_size//2) & (evt_idx<(len(data)-kernel_size//2))]
    mean_signal = np.zeros((len(evt_idx), kernel_size))
    
    for i,idx in enumerate(evt_idx): #discard the last event
        mean_signal[i,:] = data[(idx-kernel_size//2):(idx+kernel_size//2+1)].ravel()
    
    return mean_signal