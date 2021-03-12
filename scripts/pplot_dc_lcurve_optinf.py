#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script generating L-curve for optinf - driven cavity

"""
import numpy as np
from nse_opinf_poddmd.load_data import get_matrices, load_snapshots
from nse_opinf_poddmd.plotting_tools import plotting_SVD_decay, plotting_obv_vel, plotting_abs_error
from nse_opinf_poddmd.optinf_tools import deriv_approx_data, optinf_quad_svd, pod_model, optinf_linear 
import nse_opinf_poddmd.optinf_tools as oit
#from optinf_tools import optinf_quad_regularizer
from nse_opinf_poddmd.dmd_tools import dmd_model, dmdc_model, dmdquad_model, sim_dmd, \
    sim_dmdc,sim_dmdquad
from scipy.linalg import norm
from scipy.linalg import svd
import matplotlib.pyplot as plt
import matplotlib
import os
import tikzplotlib

def fab(a, b): return np.concatenate((a, b), axis=0)


###########################################################################
###### System parameters ##################################################
###########################################################################

problem = 'drivencavity'
# Ration between traning and test data
ratio = 0.8
Nprob = 2
nseodedata = False
#nseodedata = True
Re = 500
t0 = 0.
tE = 6  # 4.
# Nts = 2**12
Nts = 2**9
nsnapshots = 2**9

if problem == 'cylinderwake':
    NVdict = {1: 5812, 2: 9356, 3: 19468}
    NV = NVdict[Nprob]
    Re = 40
else:
    NVdict = {1: 722, 2: 3042, 3: 6962}
    NV = NVdict[Nprob]
    Re = 500


# Make it come true
plot_results        = True
compute_pod         = False
compute_pressure    = True


if Nprob in [1,2]: 
    tol_lstsq    = 1e-7
else:
    tol_lstsq    = 1e-7
 
tol_lstsq_dmdc   = 1e-8
###########################################################################
###### Loading system data ################################################
###########################################################################

print('Loading data for '+problem+ ' problem with NV =', NV, 'and Re =',Re)  
print('\n')

# getting system matrices    
M, A11, A12, H, B1, B2, Cv, Cp = get_matrices(problem, NV)

# loading snapshots
# V, Vd, MVd, P, T = load_snapshots(1, NV, problem, Re, 
#                                   False, False, odeint=nseodedata)
V, Vd, MVd, P, T = load_snapshots(N=Nprob, problem='drivencavity',
                                  Re=Re, tE=tE, Nts=Nts, nsnapshots=nsnapshots,
                                  odesolve=nseodedata)

# Training and test data
Vf = V                        # Vf correponds to the test velocity data
Tf = T                        # Tf correponds to the time interval for Tf
V  = Vf[:,:int(len(T)*ratio)] # V correponds to the training velocity data
T  = T[:int(len(T)*ratio)]    # T correponds to the time interval for T

###########################################################################
###### Computing reduced basis ############################################
###########################################################################

Uv, Sv, VvT = svd(V)

# plotting decay of singular values
plotting_SVD_decay(Sv)           

# order of reduced models
rv  = 30
Uvr = Uv[:,:rv]
print('The reduced basis satisfies the algebraic contrains with an error = '
      , norm(A12.T@Uvr))
print('\n')


###########################################################################
###### Computing reduced trajectories######################################
###########################################################################

dt = T[1]-T[0]
V_red = Uvr.T@V
N_sims = 0

# Computing reduced derivatives
Vd_red = deriv_approx_data(V_red, dt, N_sims)


# defining input and kron matrices
U           = np.ones((1,V_red.shape[1]))   
V_kron_red  = np.array([np.kron(V_red[:,i],V_red[:,i]) for i in range(V_red.shape[1])]).T

    
# operator inference optimization problem
X           = np.concatenate((V_red,V_kron_red, U),axis=0)
Ux, Sx, VxT = svd(X)   # Ux*Sx@VxT[:Ux.shape[0],:]-X
plotting_SVD_decay(Sx,'Optinf - lstsquares singular values')
#rx = sum(Sx/Sx[0]>tol_lstsq)

err = []
Ynorm = []
L = np.logspace(-11,-6,200)
print('Problem with regularizer - The L curve: ')
b = Vd_red.T
A1 = X.T
for i in L:
    rx = sum(Sx/Sx[0]>i)
    Y1 = Vd_red@VxT[:rx,:].T@np.diag(1/Sx[:rx])@Ux[:,:rx].T
    err.append(norm(A1@Y1.T-b))
    Ynorm.append(norm(Y1))
    print(i)
    
fig = plt.figure()
ax = plt.subplot(111)
plt.title("L-curve")
plt.scatter(err,Ynorm, c=L, norm=matplotlib.colors.LogNorm())
plt.xlim(np.min(err)*0.9,np.max(err)*1.1)
cbar = plt.colorbar()
cbar.set_label('$tol$')
plt.xlabel('least square error')
plt.ylabel('norm of solution')
ax.set_xscale('log')
ax.set_yscale('log')
plt.show()



if not os.path.exists('Figures'):
    os.makedirs('Figures')

tikzplotlib.save("Figures/driven_cavity_l_curve_3042.tex")
plt.show()
fig.savefig("Figures/driven_cavity_l_curve_3042.pdf")
