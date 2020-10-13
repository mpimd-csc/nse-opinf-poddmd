""""Tools for DMD"""

__all__ = [ 
            "dmd_model",
            "dmdc_model",
            "dmdquad_model",
            "sim_dmd",
            "sim_dmdc",
            "sim_dmdquad"
          ]

import numpy as np
from scipy.linalg import svd
from nse_opinf_poddmd.plotting_tools import plotting_SVD_decay


################################
## DMD model
################################

def dmd_model(Uvr,V,rv):
    # DMD data
    V0 = Uvr.T@V[:,:-1]
    V1 = Uvr.T@V[:,1:]
    Udmd, Sdmd, VdmdT    = svd(V0)
    
    # Computing dmd model
    Admd  = Udmd[:,:rv].T@ V1 @ VdmdT[:rv,:].T@ np.diag(1/Sdmd[:rv])
    
    return Admd

################################
## DMD model with control
################################

def dmdc_model(Uvr,V,rv,tol_lstsq = 1e-8):
    # DMD data
    V0 = Uvr.T@V[:,:-1]
    V1 = Uvr.T@V[:,1:]
    U0 = np.ones((1,V0.shape[1])) 
    Xdmdc   = np.concatenate((V0, U0),axis=0)
    
    # least square problem via svd (minimal norm solution)
    Udmdc, Sdmdc, VdmdcT = svd(Xdmdc)   # Ux*Sx@VxT[:Ux.shape[0],:]-X
    rx     = sum(Sdmdc/Sdmdc[0]>tol_lstsq)
    Ydmdc = V1@VdmdcT[:rx,:].T@np.diag(1/Sdmdc[:rx])@Udmdc[:,:rx].T
    
    # Getting dmdc model
    Admdc = Ydmdc[:,:rv]
    Bdmdc = Ydmdc[:,rv] 
    Bdmdc = Bdmdc.reshape(-1,1)
    
    return Admdc, Bdmdc 

####################################
## DMD quadratic model with control
####################################

def dmdquad_model(Uvr,V,rv, tol_lstsq = 1e-8):
    # DMD data
    V0 = Uvr.T@V[:,:-1]
    V1 = Uvr.T@V[:,1:]
    U0 = np.ones((1,V0.shape[1])) 
    V0_kron  = np.array([np.kron(V0[:,i],V0[:,i]) for i in range(V0.shape[1])]).T
    Xdmd_quad   = np.concatenate((V0, V0_kron, U0),axis=0)
    
    # least square problem via svd (minimal norm solution)
    Udmd_quad, Sdmd_quad, Vdmd_quadT = svd(Xdmd_quad)   # Ux*Sx@VxT[:Ux.shape[0],:]-X
    plotting_SVD_decay(Sdmd_quad,'DMDquad - lstsquares singular values')
    rx = sum(Sdmd_quad/Sdmd_quad[0]>tol_lstsq)
    Ydmd_quad   = V1@Vdmd_quadT[:rx,:].T@np.diag(1/Sdmd_quad[:rx])@Udmd_quad[:,:rx].T
    
    # Getting dmd quad model
    Admd_quad   = Ydmd_quad[:,:rv]
    Hdmd_quad   = Ydmd_quad[:,rv:(rv+rv**2)]
    Bdmd_quad   = Ydmd_quad[:,(rv+rv**2):]
    Bdmd_quad   = Bdmd_quad.reshape(-1,1)
    
    return Admd_quad,  Hdmd_quad, Bdmd_quad

####################################
## Simulating DMD 
####################################

def sim_dmd(Admd, x0, nb_iter):
    Vrdmd = np.empty((x0.shape[0], 0), float)
    Vrdmd = np.append(Vrdmd, x0.reshape(-1,1), axis=1)
    # Getting next step
    for t in range(nb_iter-1):
        Vrdmd = np.append(Vrdmd, Admd@Vrdmd[:,-1:], axis=1)
    
    return Vrdmd  

####################################
## Simulating DMD with control
####################################

def sim_dmdc(Admdc, Bdmdc, x0, nb_iter):
    Vrdmdc = np.empty((x0.shape[0], 0), float)
    Vrdmdc = np.append(Vrdmdc, x0.reshape(-1,1), axis=1)
    
    for t in range(nb_iter-1):
        Vrdmdc = np.append(Vrdmdc, Admdc@Vrdmdc[:,-1:]+Bdmdc, axis=1)
    return Vrdmdc

####################################
## Simulating DMD quadratic
####################################

def sim_dmdquad(Admd_quad, Hdmd_quad, Bdmd_quad, x0, nb_iter):
    Vrdmd_quad = np.empty((x0.shape[0], 0), float)
    Vrdmd_quad = np.append(Vrdmd_quad, x0.reshape(-1,1), axis=1)
    for t in range(nb_iter-1):
        Vrdmd_quad = np.append(Vrdmd_quad, Admd_quad@Vrdmd_quad[:,-1:]+ \
                           Hdmd_quad@np.kron(Vrdmd_quad[:,-1:],Vrdmd_quad[:,-1:])\
                           +Bdmd_quad, axis=1)
    return Vrdmd_quad
