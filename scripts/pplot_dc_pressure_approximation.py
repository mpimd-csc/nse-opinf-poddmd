"""
Pressure plot - Driven Cavity

"""

import numpy as np
from nse_opinf_poddmd.load_data import get_matrices, load_snapshots
from nse_opinf_poddmd.plotting_tools import plotting_SVD_decay, plotting_obv_vel, plotting_abs_error
from nse_opinf_poddmd.optinf_tools import deriv_approx_data, optinf_quad_svd
import nse_opinf_poddmd.optinf_tools as oit
#from optinf_tools import optinf_quad_regularizer
from scipy.linalg import norm
from scipy.linalg import svd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
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
plot_results        = False




if Nprob in [1,2]: 
    tol_lstsq    = 1e-8
else:
    tol_lstsq    = 1e-7
 
tol_lstsq_dmdc   = 1e-8

if nseodedata:
    tol_lstsq    = 1e-5
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

P = P[:,1:]
Vf = V
#V  = Vf[:,:int(len(T)*2/3)]

# Training and test data
Vf = V                          # Vf correponds to the test velocity data
Tf = T                          # Tf correponds to the time interval for Tf
Pf = P
V  = Vf[:,:int(len(Tf)*ratio)]    # V correponds to the training velocity data
T  = Tf[:int(len(Tf)*ratio)]      # T correponds to the time interval for T
P  = Pf[:,:int(len(Tf)*ratio)-1]  # P correpomds the training pressure data


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


P_red = Upr.T@P

###########################################################################
###### Operator inference quadratic model #################################
###########################################################################

print('Computing operator inference model... \n')


Aoptinf, Hoptinf, Boptinf = optinf_quad_svd(V_red, Vd_red, tol_lstsq)
    

###########################################################################
###### Operator inference pressure model  #################################
###########################################################################
V_red_p       = V_red[:,1:]
Vd_red_p      = Vd_red[:,1:]
U_p           = np.ones((1,V_red_p.shape[1]))   
V_kron_red_p  = np.array([np.kron(V_red_p[:,i],V_red_p[:,i]) for i in range(V_red_p.shape[1])]).T

X_p          = np.concatenate((V_red_p,V_kron_red_p, U_p),axis=0)
Ux_p, Sx_p, VxT_p = svd(X_p)   # Ux*Sx@VxT[:Ux.shape[0],:]-X
plotting_SVD_decay(Sx_p,'Optinf - lstsquares singular values')
rx = sum(Sx_p/Sx_p[0]>tol_lstsq)

Ypsvd = P_red@VxT_p[:rx,:].T@np.diag(1/Sx_p[:rx])@Ux_p[:,:rx].T

Ap      = Ypsvd[:,0:rv]
Hp      = Ypsvd[:,rv:(rv+rv**2)]
Bp      = Ypsvd[:,(rv+rv**2):]
Bp = np.squeeze(Bp,axis=1)
###########################################################################
###### Simulatind systems #################################################
###########################################################################

print('Simulating reduced order systems... \n')

# projected initial condition
x0 = Uvr.T@V[:,0]

# simulating Optinf model
optinf_qm = oit.get_quad_model(A=Aoptinf, H=Hoptinf, B=Boptinf)
xsol_optinf     = odeint(optinf_qm, x0, Tf)  # , args=(Aoptinf,Hoptinf,Boptinf))
Voptinf         = Uvr @ xsol_optinf.T 
#V_kron_optinf  = np.array([np.kron(Voptinf[:,i],Voptinf[:,i]) for i in range(Voptinf.shape[1])]).T
Vr_optinf       = xsol_optinf.T 
Vr_optinf       = Vr_optinf[:,1:] 
V_kron_r_optinf = np.array([np.kron(Vr_optinf[:,i],Vr_optinf [:,i]) for i in range(Vr_optinf.shape[1])]).T

###########################################################################
###### Plotting Pressure Results ##########################################
###########################################################################

plotting_abs_error(Vf, Voptinf, Tf, 'Optinf' )


Proptinf    = Ap@Vr_optinf +Hp@V_kron_r_optinf + Bp.reshape(-1,1)
Poptinf     = Upr@Proptinf
Tp          = Tf[1:] 

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, tight_layout=True)

ax1.plot(Tp, Cp[:, :-1].dot(Pf).T)
ax1.set_ylabel('$y_p(t) = C_p p(t)$')
ax1.set_title('Full Order Model')

ax2.plot(Tp, Cp[:, :-1].dot(Poptinf).T)
ax2.set_ylabel('$\\hat y_p(t) = C_p \\hat p(t)$')
ax2.set_title('Operator Inference')

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

ax3.semilogy(Tp, abs(Cp[:, :-1].dot(Pf-Poptinf).T), color=colors[1])
ax3.yaxis.set_ticks([1e-4,1e-7,1e-10])
ax3.set_xlabel('time $t$')
ax3.set_ylabel('$|y_p(t)-\\hat y_p(t)|$')
ax3.set_title('Approximation Error')
ax1.axvline(x=T[-1], color='k', linestyle='--')
ax2.axvline(x=T[-1], color='k', linestyle='--') 
ax3.axvline(x=T[-1], color='k', linestyle='--') 
    
print('\nOptinf pressure error: ', norm(Poptinf-Pf))
tikzplotlib.save("Figures/driven_cavity_pressure_3042.tex")
plt.show()
fig.savefig("Figures/driven_cavity_pressure_3042.pdf")
# Take 2**9 snapshots, Tend = 6, rv = 30 
#
#
plt.show()
