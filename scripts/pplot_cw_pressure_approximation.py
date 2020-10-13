"""
Pressure plot - Cylinder wake

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
compute_pod         = False
compute_pressure    = True

if Re  == 40: 
    tol_lstsq    = 1e-7
else:
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

V, Vd, MVd, P, T = load_snapshots(N=Nprob, problem='cylinderwake',
                                  Re=Re, tE=tE, Nts=Nts, nsnapshots=nsnapshots,
                                  odesolve=nseodedata)

print('Number of snapshot: ',len(T))
print('Time span: ',T[-1])

P = P[:,1:]
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
xsol_optinf     = odeint(optinf_qm, x0, T)  # , args=(Aoptinf,Hoptinf,Boptinf))
Voptinf         = Uvr @ xsol_optinf.T 




###########################################################################
###### Plotting Pressure Results ##########################################
###########################################################################

plotting_abs_error(Vf, Voptinf, T, 'Optinf' )




Proptinf    = Ap@V_red_p +Hp@V_kron_red_p + Bp.reshape(-1,1)
Poptinf     = Upr@Proptinf
Tp          = T[1:] 

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, tight_layout=True)

ax1.plot(Tp, Cp.dot(P).T)
ax1.set_ylabel('$y_p(t) = C_p p(t)$')
ax1.set_title('Full Order Model')

ax2.plot(Tp, Cp.dot(Poptinf).T)
ax2.set_ylabel('$\\hat y_p(t) = C_p \\hat p(t)$')
ax2.set_title('Operator Inference')

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

ax3.plot(Tp, Cp.dot(P-Poptinf).T, color=colors[1])
ax3.set_xlabel('time $t$')
ax3.set_ylabel('$y_p(t)-\\hat y_p(t)$')
ax3.set_title('Approximation Error')

print('\nOptinf pressure error: ', norm(Poptinf-P))

tikzplotlib.save("./Figures/cylinder_wake_pressure.tex")
fig.savefig("./Figures/cylinder_wake_pressure.pdf")

plt.show(block=False)
