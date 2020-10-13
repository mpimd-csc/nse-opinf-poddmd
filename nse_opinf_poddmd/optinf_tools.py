""""Tools for operator inference"""

__all__ = [
            "pod_model",
            "get_quad_model",
            "deriv_approx_data",
            "optinf_quad_svd",
            "optinf_quad_regularizer",
            "optinf_linear",
            "lin_model"
          ]

from nse_opinf_poddmd.Approx_derivative import xdot
from nse_opinf_poddmd.plotting_tools import plotting_SVD_decay

import numpy as np
from scipy.linalg import svd, norm
from numpy.linalg import solve
import matplotlib.pyplot as plt

################################
## POD model
################################


def pod_model(Uvr, M, A11, H, B1, ret_hpodfunc=False):

    # Computing POD model 
    Mr     = Uvr.T@M@Uvr
    MrinvUvrt = solve(Mr, Uvr.T)
    Apod   = MrinvUvrt@A11@Uvr
    Hpod   = solve(Mr, Uvr.T)@H  # No mutiplication on the right with kron(Uvr, Uvr)
    # better keep the sparsity of H as long as possible

    def Hpodfunc(vr):
        fullv = Uvr.dot(vr)
        Hfvfv = H.dot(np.kron(fullv, fullv))
        return MrinvUvrt.dot(Hfvfv)

    Bpod   = MrinvUvrt@B1
    Bpod   = np.squeeze(Bpod, axis=1)

    if ret_hpodfunc:
        return Apod, Hpod, Bpod, Hpodfunc
    else:
        return Apod, Hpod, Bpod


################################
# Get the Quadratic Models
################################

def get_quad_model(A=None, H=None, Hfunc=None, B=None, podbase=None):

    if H is not None:
        if podbase is None:
            def quad_model(x, t):
                dxdt = A @ x + H @ np.kron(x, x) + B
                return dxdt
        else:
            def quad_model(x, t):
                dxdt = A @ x + H @ np.kron(podbase@x, podbase@x) + B
                return dxdt

    elif Hfunc is not None:
        def quad_model(x, t):
            dxdt = A @ x + Hfunc(x) + B
            return dxdt

    return quad_model


################################
## Derivative approximation
################################

def deriv_approx_data(V_red, dt, N_sims=0):
    Vd_red_approx = np.zeros_like(V_red)
    L = np.int(V_red.shape[1]/(N_sims+1))

    for iter_x0 in range(N_sims+1):
        Vd_red_approx[:,iter_x0*L:(iter_x0+1)*L] = xdot(V_red[:,iter_x0*L:(iter_x0+1)*L], dt, order=6)
    
    return Vd_red_approx

################################
## Opt inf svd
################################

def optinf_quad_svd(V_red, Vd_red, tol_lstsq = 1e-8, P_red = None):
    # defining input and kron matrices
    U           = np.ones((1,V_red.shape[1]))   
    V_kron_red  = np.array([np.kron(V_red[:,i],V_red[:,i]) for i in range(V_red.shape[1])]).T
    rv          = V_red.shape[0]
    
    # operator inference optimization problem
    X           = np.concatenate((V_red,V_kron_red, U),axis=0)
    Ux, Sx, VxT = svd(X)   # Ux*Sx@VxT[:Ux.shape[0],:]-X
    plotting_SVD_decay(Sx,'Optinf - lstsquares singular values')
    rx = sum(Sx/Sx[0]>tol_lstsq)
    
    # computing minimal norm SVD solution for velocity
    Ysvd = Vd_red@VxT[:rx,:].T@np.diag(1/Sx[:rx])@Ux[:,:rx].T
    print('error norm(Ysvd@X-b.T) for svd = ', norm(Ysvd@X-Vd_red)/norm(Ysvd))
    print('norm of the least squares solution Ysvd = ', norm(Ysvd))
    print('\n')
    
    # getting optinf matrices
    Aoptinf     = Ysvd[:,0:rv]
    Hoptinf     = Ysvd[:,rv:(rv+rv**2)]
    Boptinf     = Ysvd[:,(rv+rv**2):]
    Boptinf     = np.squeeze(Boptinf,axis=1)  

    if P_red is None:
        return  Aoptinf, Hoptinf, Boptinf
    
    else: 
        # computing minimal norm SVD solution for pressure
        Ypsvd = P_red@VxT[:rx,:].T@np.diag(1/Sx[:rx])@Ux[:,:rx].T
        print('error norm(Ypressure_svd@X-b) for svd = ', norm(Ypsvd@X-P_red))
        print('norm of the least squares solution Ysvd = ', norm(Ypsvd))
        print('\n')
    
        # getting optinf matrices
        Ap      = Ypsvd[:,0:rv]
        Hp      = Ypsvd[:,rv:(rv+rv**2)]
        Bp      = Ypsvd[:,(rv+rv**2):]
        Bp = np.squeeze(Bp,axis=1)
    
        return Aoptinf, Hoptinf, Boptinf, Ap, Hp, Bp    
    

################################
## Opt inf regulatizer
################################

def optinf_quad_regularizer(V_red, Vd_red):
    # defining input and kron matrices
    U           = np.ones((1,V_red.shape[1]))   
    V_kron_red  = np.array([np.kron(V_red[:,i],V_red[:,i]) for i in range(V_red.shape[1])]).T
    rv          = V_red.shape[0]
    
    # operator inference optimization problem
    X           = np.concatenate((V_red,V_kron_red, U),axis=0)
    b = Vd_red.T
    A1 = X.T
    
    # defining regulirez problems
    A1tb = A1.T@b
    A1tA1 = A1.T@A1
    sol = lambda l: solve(A1tA1 + l*np.eye(rv + rv**2 + 1), A1tb)

    err = []
    Ynorm = []
    L = np.logspace(-12,-4,50)
    print('Problem with regularizer - The L curve: ')
    for i in L:
        Y1 = sol(i)
        err.append(norm(A1@Y1-b))
        Ynorm.append(norm(Y1))
        print(i)
    
    fig = plt.figure()
    plt.scatter(err,Ynorm, c=L)
    plt.xlim(np.min(err)*0.9,np.max(err)*1.1)
    plt.colorbar()
    plt.show()

    Y = sol(L[10]).T

    # getting optinf matrices
    Aoptinf     = Y[:,0:rv]
    Hoptinf     = Y[:,rv:(rv+rv**2)]
    Boptinf     = Y[:,(rv+rv**2):]
    Boptinf     = np.squeeze(Boptinf,axis=1)
    
    return Aoptinf, Hoptinf, Boptinf

################################
## Opt inf: Pressure with svd
################################

def optinf_pressure_svd(P_red, V_red):
    # defining input and kron matrices
    U           = np.ones((1,V_red.shape[1]))   
    V_kron_red  = np.array([np.kron(V_red[:,i],V_red[:,i]) for i in range(V_red.shape[1])]).T
    rv          = V_red.shape[0]

    # operator inference optimization problem
    X           = np.concatenate((V_red,V_kron_red, U),axis=0)
    Ux, Sx, VxT = svd(X)   # Ux*Sx@VxT[:Ux.shape[0],:]-X
    rx = sum(Sx/Sx[0]>1e-8)
    
    # computing minimal norm SVD solution
    Ypsvd = P_red@VxT[:rx,:].T@np.diag(1/Sx[:rx])@Ux[:,:rx].T
    print('error norm(Ypressure_svd@X-b) for svd = ', norm(Ypsvd@X-P_red))
    print('norm of the least squares solution Ysvd = ', norm(Ypsvd))
    print('\n')
    
    # getting optinf matrices
    Ap      = Ypsvd[:,0:rv]
    Hp      = Ypsvd[:,rv:(rv+rv**2)]
    Bp      = Ypsvd[:,(rv+rv**2):]
    Bp = np.squeeze(Bp,axis=1)
    
    return Ap, Hp, Bp

#####################################
## Opt inf: Pressure with regularizer
####################################

def optinf_pressure_regularizer(P_red, V_red):
    # defining input and kron matrices
    U           = np.ones((1,V_red.shape[1]))   
    V_kron_red  = np.array([np.kron(V_red[:,i],V_red[:,i]) for i in range(V_red.shape[1])]).T
    rv          = V_red.shape[0]

    # operator inference optimization problem
    X           = np.concatenate((V_red,V_kron_red, U),axis=0)
    
    bp = P_red.T
    A1 = X.T

    A1tbp = A1.T@bp
    A1tA1 = A1.T@A1
    sol_p = lambda l: solve(A1tA1 + l*np.eye(rv + rv**2 + 1), A1tbp)


    Yp1 = sol_p(1e-8).T
    p_opInf_red = Yp1@X

    return p_opInf_red


#####################################
## Opt inf for linear model
####################################

def optinf_linear(V_red, Vd_red):
    # defining input and kron matrices
    U           = np.ones((1,V_red.shape[1]))   
    rv          = V_red.shape[0]
    
    # operator inference optimization problem
    X           = np.concatenate((V_red, U),axis=0)
    Ux, Sx, VxT = svd(X)   # Ux*Sx@VxT[:Ux.shape[0],:]-X
    plotting_SVD_decay(Sx,'Optinf - lstsquares singular values')
    rx = sum(Sx/Sx[0]>1e-6)
    
    # computing minimal norm SVD solution for velocity
    Ysvd = Vd_red@VxT[:rx,:].T@np.diag(1/Sx[:rx])@Ux[:,:rx].T
    print('error norm(Ysvd@X-b.T) for svd = ', norm(Ysvd@X-Vd_red)/norm(Ysvd))
    print('norm of the least squares solution Ysvd = ', norm(Ysvd))
    print('\n')
    
    # getting optinf matrices
    Aoptinf     = Ysvd[:,0:rv]
    Boptinf     = Ysvd[:,rv:]
    Boptinf     = np.squeeze(Boptinf,axis=1) 
    
    return Aoptinf, Boptinf

def lin_model(x, t, A, B):
    dxdt = A @ x + B
    return dxdt
