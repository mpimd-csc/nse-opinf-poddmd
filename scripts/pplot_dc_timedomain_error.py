from nse_opinf_poddmd.load_data import get_matrices, load_snapshots
from nse_opinf_poddmd.plotting_tools import plotting_SVD_decay, plotting_obv_vel, plotting_abs_error
from nse_opinf_poddmd.optinf_tools import deriv_approx_data, optinf_quad_svd, pod_model, optinf_linear 
import nse_opinf_poddmd.optinf_tools as oit
#from optinf_tools import optinf_quad_regularizer
from nse_opinf_poddmd.dmd_tools import dmd_model, dmdc_model, dmdquad_model, sim_dmd, \
    sim_dmdc,sim_dmdquad

import numpy as np
from scipy.linalg import norm
from scipy.linalg import svd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os
import tikzplotlib

def fab(a, b): return np.concatenate((a, b), axis=0)


###########################################################################
###### System parameters ##################################################
###########################################################################

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
plot_results        = False
compute_pod         = True
compute_pressure    = False


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

Vf = V
#V  = Vf[:,:int(len(T)*2/3)]

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

# Reduced basis for pressure
if compute_pressure:
    Up, Sp, VpT = svd(P)
    
    # plotting decay of singular values
    plotting_SVD_decay(Sp, 'pressure')      
    
    rp  = 30
    Upr = Up[:,:rp]


###########################################################################
###### Computing reduced trajectories######################################
###########################################################################

dt = T[1]-T[0]
V_red = Uvr.T@V
N_sims = 0

# Computing reduced derivatives
Vd_red = deriv_approx_data(V_red, dt, N_sims)

if compute_pressure:
    P_red = Upr.T@P

###########################################################################
###### Operator inference quadratic model #################################
###########################################################################

print('Computing operator inference model... \n')

if compute_pressure:
    
    Aoptinf, Hoptinf, Boptinf, Ap, Hp, Bp  = optinf_quad_svd(V_red, Vd_red, tol_lstsq, P_red)
      
else: 
    Aoptinf, Hoptinf, Boptinf = optinf_quad_svd(V_red, Vd_red, tol_lstsq)
    

###########################################################################
###### Operator inference linear model ####################################
###########################################################################

Aoptinf_lin, Boptinf_lin = optinf_linear(V_red, Vd_red)

###########################################################################
###### POD quadratic model ################################################
###########################################################################

if compute_pod:
    print('Computing POD model... \n')
    Apod, Hpod, Bpod, Hpodfunc = pod_model(Uvr, M, A11, H, B1,
                                           ret_hpodfunc=True)

###########################################################################
###### DMD  model #########################################################
###########################################################################

print('Computing DMD models... \n')
Admd = dmd_model(Uvr,V,rv)

###########################################################################
###### DMD model with control #############################################
###########################################################################

Admdc, Bdmdc = dmdc_model(Uvr,V,rv, tol_lstsq_dmdc)
  
###########################################################################
###### DMD quadratic model with control ###################################
###########################################################################

Admd_quad, Hdmd_quad, Bdmd_quad = dmdquad_model(Uvr,V,rv)

###########################################################################
###### Simulatind systems #################################################
###########################################################################

print('Simulating reduced order systems... \n')

# projected initial condition
x0 = Uvr.T@V[:,0]

# simulating Optinf model
optinf_qm = oit.get_quad_model(A=Aoptinf, H=Hoptinf, B=Boptinf)
xsol_optinf     = odeint(optinf_qm, x0, T)  # , args=(Aoptinf,Hoptinf,Boptinf))
Voptinf         = Uvr @ xsol_optinf.T 

# simulatinf OptInf linear model
xsol_optinf_lin     = odeint(oit.lin_model, x0, T, (Aoptinf_lin, Boptinf_lin))
Voptinf_lin         = Uvr @ xsol_optinf_lin.T 

# simulating POD model
if compute_pod:
    print('POD...')
    # pod_qm = oit.get_quad_model(A=Apod, H=Hpod, B=Bpod, podbase=Uvr)
    pod_qm = oit.get_quad_model(A=Apod, Hfunc=Hpodfunc, B=Bpod)
    xsol_pod    = odeint(pod_qm, x0, T)  # args=(Apod,Hpod,Bpod))
    Vpod        = Uvr @ xsol_pod.T  

# simulating DMD model
Vrdmd = sim_dmd(Admd, x0, len(T))
Vdmd = Uvr@Vrdmd

# Simulating DMD model with control 
Vrdmdc =sim_dmdc(Admdc, Bdmdc, x0, len(T))
Vdmdc = Uvr@Vrdmdc

# Simulating DMD quadratic model with control 
#Vrdmd_quad = sim_dmdquad(Admd_quad, Hdmd_quad, Bdmd_quad, x0, len(T))
#Vdmd_quad  = Uvr@Vrdmd_quad


###########################################################################
###### Plotting results ###################################################
###########################################################################



fig = plt.figure()
# Time domain simulation for some observed trajectories
ax = plt.subplot(121)
ax.plot(T, (Cv@V).T, 'k')
ax.plot(T,(Cv@V).T[:,0], 'k', label='FOM')
ax.plot(T, (Cv@Voptinf).T, '--b')
ax.plot(T,(Cv@Voptinf).T[:,0],'--b', label='OpInf')
ax.plot(T, (Cv@Voptinf_lin ).T,'c--')
ax.plot(T,(Cv@Voptinf_lin ).T[:,0],'c--', label = 'OpInf linear')
ax.plot(T, (Cv@Vpod).T, 'm--')
ax.plot(T,(Cv@Vpod).T[:,0],'m--', label='POD')
ax.plot(T, (Cv@Vdmd).T,'r--')
ax.plot(T,(Cv@Vdmd).T[:,0],'r--', label = 'DMD')
ax.plot(T, (Cv@Vdmdc).T,'g--')
ax.plot(T,(Cv@Vdmdc).T[:,0],'g--', label = 'DMDc')
plt.legend(loc='best')
plt.xlabel('time (sec)')
plt.ylabel('y')
plt.legend()
plt.legend(loc='upper right')
ax.set_title("Time-domain simulation")

# Mean error for different methods
ax = plt.subplot(122)
ax.semilogy(T,np.mean(np.abs(V - Voptinf).T,axis=1), '--b', label='OpInf')
ax.semilogy(T,np.mean(np.abs(V - Voptinf_lin).T,axis=1), '--c', label='OpInf linear')
ax.semilogy(T,np.mean(np.abs(V - Vpod).T,axis=1), 'm--', label='POD')
ax.semilogy(T,np.mean(np.abs(V - Vdmd).T,axis=1), 'r--', label='DMD')
ax.semilogy(T,np.mean(np.abs(V - Vdmdc).T,axis=1), 'g--', label='DMDc')
plt.xlabel('time (sec)')
plt.ylabel('$L_{\infty}$ error')
plt.legend()
plt.legend(loc='upper right')
ax.set_title("Approximation error")
plt.subplots_adjust(wspace = 0.5)

if not os.path.exists('Figures'):
    os.makedirs('Figures')

tikzplotlib.save("./Figures/driven_cavity_time_domain_3042.tex")
plt.show()
fig.savefig("./Figures/driven_cavity_time_domain_3042.pdf")    

if compute_pod:
    print('POD error: ', norm(Vpod-Vf))

print('Optinf error: ', norm(Voptinf-Vf))
print('Optinf lin error: ', norm(Voptinf_lin-Vf))
print('DMD error: ', norm(Vdmd-Vf))
print('DMDc error: ', norm(Vdmdc-Vf))
