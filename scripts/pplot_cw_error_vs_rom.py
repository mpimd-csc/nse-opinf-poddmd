
"""
Script generating the error decay for different orders - Cylinder wake

"""

import numpy as np
from nse_opinf_poddmd.load_data import get_matrices, load_snapshots
from nse_opinf_poddmd.plotting_tools import plotting_SVD_decay, plotting_obv_vel, plotting_abs_error
from nse_opinf_poddmd.optinf_tools import deriv_approx_data, optinf_quad_svd, pod_model, optinf_linear
import nse_opinf_poddmd.optinf_tools as oit
#from optinf_tools import optinf_quad_regularizer
from nse_opinf_poddmd.dmd_tools import dmd_model, dmdc_model,  sim_dmd, \
    sim_dmdc
from scipy.linalg import norm
from scipy.linalg import svd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.linalg import solve
import scipy as sp
import os
import tikzplotlib


def fab(a, b): return np.concatenate((a, b), axis=0)

###########################################################################
###### System parameters ##################################################
###########################################################################

problem = 'cylinderwake'
Nprob = 1
nseodedata = False

tE = 2  # 0.4
Nts = 2**9
nsnapshots = 2**9

if problem == 'cylinderwake':
    NVdict = {1: 5812, 2: 9356, 3: 19468}
    NV = NVdict[Nprob]
    Re = 60 #60 
else:
    NVdict = {1: 722, 2: 3042, 3: 6962}
    NV = NVdict[Nprob]
    Re = 500

plot_results        = True
compute_pod         = True
compute_pressure    = False

if Re  == 40: 
    tol_lstsq    = 1e-7
else:
    tol_lstsq    = 1e-5
    
# Make it come true
plot_results        = True
compute_pod         = True
compute_pressure    = False

# Range of desired reduced orders to be computed
rv_init  = 6
rv_final = 32
rv_step  = 2


# Error vectors
err_optinf      =  []
err_optinf_lin  =  []
err_pod         =  []
err_dmd         =  []
err_dmdc        =  []

if Nprob in [1,2]: 
    tol_lstsq    = 1e-7
else:
    tol_lstsq    = 1e-7
 
tol_lstsq_dmdc   = 1e-8

print('Loading data for '+problem+ ' problem with NV =', NV, 'and Re =',Re)  
print('\n')



# getting system matrices    
M, A11, A12, H, B1, B2, Cv, Cp = get_matrices(problem, NV)

# loading snapshots
# V, Vd, MVd, P, T = load_snapshots(1, NV, problem, Re,
#                                   False, False, odeint=nseodedata)

V, Vd, MVd, P, T = load_snapshots(N=Nprob, problem='cylinderwake',
                                  Re=Re, tE=tE, Nts=Nts, nsnapshots=nsnapshots,
                                  odesolve=nseodedata)

print('Number of snapshot: ',len(T))
print('Time span: ',T[-1])

# tranformation to divergent free system 

if compute_pod:
    # divergency free velocity
    B2  = -A12.T@V[:,0]                      # => A12.T@V+B2 = 0 
    A12 = A12.todense()
    Cst = np.array(-A12@ solve((A12.T@A12), B2.reshape(-1,1)))
    V_divfree = V-Cst
    print('The diverence free velocity satisfies the algebraic contrains with an error = '
      , norm(A12.T@V_divfree))
    
    # Shifted system as V = V_divfree + Cst
    Id = sp.sparse.identity(NV)
    Adivfree = A11 + H@sp.sparse.kron(Id,Cst) + H@sp.sparse.kron(Cst,Id)
    Hdivfree = H
    Bdivfree = B1 + A11@Cst +H@(np.kron(Cst,Cst)) 
    Bdivfree = np.array(Bdivfree)
    Uv_divfree, Sv_divfree, VvT_divfree = svd(V_divfree)

Vf = V
V = Vf

###########################################################################
###### Computing reduced basis ############################################
###########################################################################

Uv, Sv, VvT = svd(V)

# plotting decay of singular values
plotting_SVD_decay(Sv)   


for rv in range(rv_init,rv_final+1,rv_step):
    print('\n\n\n')
    print('Computation for rv =', rv)
    print('\n')
    Uvr = Uv[:,:rv]
    print('The reduced basis satisfies the algebraic contrains with an error = '
      , norm(A12.T@Uvr))
    print('\n')
    
    tol_lstsq  = 1e-5
    # Operator inference    
    dt = T[1]-T[0]
    V_red = Uvr.T@V
    N_sims = 0
    Vd_red = deriv_approx_data(V_red, dt, N_sims)
    Aoptinf, Hoptinf, Boptinf = optinf_quad_svd(V_red, Vd_red, tol_lstsq)
    
    # Operator inference linear
    Aoptinf_lin, Boptinf_lin = optinf_linear(V_red, Vd_red)
    
    
    # POD
    if compute_pod:
        Uv_divfree, Sv_divfree, VvT_divfree = svd(V_divfree)
        Uvr_divfree = Uv_divfree[:,:rv]
        print('The diverence free reduced basis satisfies the algebraic contrains with an error = '
              , norm(A12.T@Uvr_divfree))
    
    
        print('Computing POD model... \n')
        Apod, Hpod, Bpod, Hpodfunc = pod_model(Uvr_divfree, M, Adivfree, Hdivfree,
                                           Bdivfree, ret_hpodfunc=True)

    # DMD
    Admd = dmd_model(Uvr,V,rv)
    
    # DMDc 
    Admdc, Bdmdc = dmdc_model(Uvr,V,rv, tol_lstsq_dmdc)
    
    # Simulation
    # projected initial condition
    x0 = Uvr.T@V[:,0]

    # simulating Optinf model
    optinf_qm = oit.get_quad_model(A=Aoptinf, H=Hoptinf, B=Boptinf)
    xsol_optinf     = odeint(optinf_qm, x0, T)  # , args=(Aoptinf,Hoptinf,Boptinf))
    Voptinf         = Uvr @ xsol_optinf.T 
    err_optinf.append(norm(Voptinf-V)*dt)
    
    # simulatinf OptInf linear model
    xsol_optinf_lin     = odeint(oit.lin_model, x0, T, (Aoptinf_lin, Boptinf_lin))
    Voptinf_lin         = Uvr @ xsol_optinf_lin.T 
    err_optinf_lin.append(norm(Voptinf_lin -V)*dt)
    # simulation POD
    if compute_pod:
        print('POD ...')
        pod_qm = oit.get_quad_model(A=Apod, Hfunc=Hpodfunc, B=Bpod)
        x0divfree   =  Uvr_divfree.T@V_divfree[:,0].flatten()
        xsol_pod    = odeint(pod_qm, x0divfree, T)  # args=(Apod,Hpod,Bpod))
        Vpod        = Uvr_divfree @ xsol_pod.T  + Cst
        err_pod.append(norm(Vpod-V)*dt)
        
    # simulating DMD model
    Vrdmd = sim_dmd(Admd, x0, len(T))
    Vdmd = Uvr@Vrdmd
    err_dmd.append(norm(Vdmd -V)*dt)
    
    # Simulating DMD model with control 
    Vrdmdc =sim_dmdc(Admdc, Bdmdc, x0, len(T))
    Vdmdc = Uvr@Vrdmdc
    err_dmdc.append(norm(Vdmdc -V)*dt)
    
range_rv = list(range(rv_init,rv_final+1,rv_step))
fig = plt.figure()
ax = plt.subplot(111)
ax.semilogy(range_rv,err_optinf,label='OpInf')
ax.semilogy(range_rv,err_optinf_lin,'c:',label='OpInf linear')
ax.semilogy(range_rv,err_dmd,'r-.',label='DMD')
ax.semilogy(range_rv,err_dmdc,'g--',label='DMDc')
if compute_pod:
    ax.semilogy(range_rv,err_pod,'m-*',label='POD')
plt.xlabel('Reduced order $r$')
plt.ylabel('$L_2$ error')
ax.legend()
ax.set_title("$L_2$ error decay - Cylinder wake")

if not os.path.exists('Figures'):
    os.makedirs('Figures')

tikzplotlib.save("./Figures/cylinder_err_vs_order.tex")
plt.show()
fig.savefig("./Figures/cylinder_err_vs_order.pdf")
