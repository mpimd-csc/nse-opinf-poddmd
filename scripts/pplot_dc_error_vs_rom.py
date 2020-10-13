
"""
Script generating the error decay for different orders - Driven cavity model

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
from scipy.integrate import odeint
import tikzplotlib
import os

problem = 'drivencavity'
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
compute_pod         = True
compute_pressure    = False

# Range of desired reduced orders to be computed
rv_init  = 6
rv_final = 30
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
V, Vd, MVd, P, T = load_snapshots(N=Nprob, problem='drivencavity',
                                  Re=Re, tE=tE, Nts=Nts, nsnapshots=nsnapshots,
                                  odesolve=nseodedata)

Vf = V
#V  = Vf[:,:int(len(T)*2/3)]

###########################################################################
###### Computing reduced basis ############################################
###########################################################################

Uv, Sv, VvT = svd(V)

# plotting decay of singular values
plotting_SVD_decay(Sv)

for rv in range(rv_init,rv_final+1,rv_step):
    print('Computation for rv =', rv)
    print('\n\n\n')
    Uvr = Uv[:,:rv]
    print('The reduced basis satisfies the algebraic contrains with an error = '
      , norm(A12.T@Uvr))
    print('\n')
    if rv <10:
        tol_lstsq  = 2*1e-4
    elif rv in [10,12,14]:
        tol_lstsq  = 1e-4
    else:
        tol_lstsq  = 1e-7
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
        Apod, Hpod, Bpod, Hpodfunc = pod_model(Uvr, M, A11, H, B1,
                                           ret_hpodfunc=True)

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
        pod_qm = oit.get_quad_model(A=Apod, Hfunc=Hpodfunc, B=Bpod)
        xsol_pod    = odeint(pod_qm, x0, T)  # args=(Apod,Hpod,Bpod))
        Vpod        = Uvr @ xsol_pod.T  
        err_pod.append(norm(Vpod -V)*dt)
        
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
ax.set_title("$L_2$ error decay - Driven cavity")

if not os.path.exists('Figures'):
    os.makedirs('Figures')

tikzplotlib.save("./Figures/driven_cavity_err_vs_order_3042.tex")
plt.show()
fig.savefig("./Figures/driven_cavity_err_vs_order_3042.pdf")
