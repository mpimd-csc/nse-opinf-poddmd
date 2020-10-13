
"""
Script generating the time-domain simulation for the cylinder wake model

"""

import numpy as np
from nse_opinf_poddmd.load_data import get_matrices, load_snapshots
from nse_opinf_poddmd.plotting_tools import plotting_SVD_decay, plotting_obv_vel, plotting_abs_error
from nse_opinf_poddmd.optinf_tools import deriv_approx_data, optinf_quad_svd, pod_model, optinf_linear
import nse_opinf_poddmd.optinf_tools as oit
#from optinf_tools import optinf_quad_regularizer
from nse_opinf_poddmd.dmd_tools import dmd_model, dmdc_model, dmdquad_model, sim_dmd, \
    sim_dmdc, sim_dmdquad
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

Vf = V
V = Vf
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
    Uv_divfree, Sv_divfree, VvT_divfree = svd(V_divfree)
    Uvr_divfree = Uv_divfree[:,:rv]
    print('The diverence free reduced basis satisfies the algebraic contrains with an error = '
      , norm(A12.T@Uvr_divfree))
    
    
    print('Computing POD model... \n')
    Apod, Hpod, Bpod, Hpodfunc = pod_model(Uvr_divfree, M, Adivfree, Hdivfree,
                                           Bdivfree, ret_hpodfunc=True)

###########################################################################
###### DMD  model #########################################################
###########################################################################

print('Computing DMD models... \n')
Admd = dmd_model(Uvr,V,rv)

###########################################################################
###### DMD model with control #############################################
###########################################################################

Admdc, Bdmdc = dmdc_model(Uvr,V,rv)
  
###########################################################################
###### DMD quadratic model with control ###################################
###########################################################################

Admd_quad, Hdmd_quad, Bdmd_quad = dmdquad_model(Uvr,V,rv,tol_lstsq)

###########################################################################
###### Simulatind systems #################################################
###########################################################################

print('Simulating reduced order systems... \n')

# projected initial condition
x0 = Uvr.T@V[:,0]

# simulating Optinf model
optinf_qm = oit.get_quad_model(A=Aoptinf, H=Hoptinf, B=Boptinf)
xsol_optinf     = odeint(optinf_qm, x0, T)
# xsol_optinf     = odeint(quad_model, x0, T, args=(Aoptinf,Hoptinf,Boptinf))
Voptinf         = Uvr @ xsol_optinf.T 

# simulatinf OptInf linear model
xsol_optinf_lin     = odeint(oit.lin_model, x0, T, (Aoptinf_lin, Boptinf_lin))
Voptinf_lin         = Uvr @ xsol_optinf_lin.T 

# simulating POD model
if compute_pod:
    print('POD ...')
    pod_qm = oit.get_quad_model(A=Apod, Hfunc=Hpodfunc, B=Bpod)
    x0divfree   =  Uvr_divfree.T@V_divfree[:,0].flatten()
    xsol_pod    = odeint(pod_qm, x0divfree, T)  # args=(Apod,Hpod,Bpod))
    Vpod        = Uvr_divfree @ xsol_pod.T  + Cst

# simulating DMD model
print('DMD ...')
Vrdmd = sim_dmd(Admd, x0, len(T))
Vdmd = Uvr@Vrdmd

# Simulating DMD model with control 
print('DMDc ...')
Vrdmdc =sim_dmdc(Admdc, Bdmdc, x0, len(T))
Vdmdc = Uvr@Vrdmdc

# Simulating DMD quadratic model with control 
print('DMDq ...')
Vrdmd_quad = sim_dmdquad(Admd_quad, Hdmd_quad, Bdmd_quad, x0, len(T))
Vdmd_quad  = Uvr@Vrdmd_quad



###########################################################################
###### Plotting results ###################################################
###########################################################################

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, tight_layout=True)
flngth = 25
tfilter = np.arange(0, len(T), flngth)
fskip = 4
trange = np.array(T)


def incrmntfilter(ctf):
    ctf = np.r_[0, ctf+fskip]
    try:
        ctr = trange[ctf]
    except IndexError:
        ctf = ctf[:-1]
        ctr = trange[ctf]
    return ctf, ctr


markerlst = ['v:', '^:', '<:', '>:', 'd:']
markerlst = ['o-', 's-', 'd-', 'D-', 'p-']
msize = 3
lw = .5
# ax = plt.subplot(212)

ctf = tfilter
print('ctf: ', ctf[1])
gtr = trange
ctr = gtr[tfilter]
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
datalist = [Voptinf, Voptinf_lin, Vpod, Vdmd, Vdmdc]
labellist = ['OpInf', 'OpInf linear', 'POD', 'DMD', 'DMDc']

ax1.plot(T, (Cv@V).T[:, 0], 'k', label='FOM')
ax1.plot(T, (Cv@V).T[:, 1:], 'k', linewidth=lw)

for kkk in range(len(datalist)):
    cmkr, ccl = markerlst[kkk], colors[kkk]
    ax2.semilogy(ctr, np.max(np.abs(V - datalist[kkk]).T, axis=1)[ctf],
                 cmkr, color=ccl, label=labellist[kkk],
                 linewidth=lw, markersize=msize)
    ax1.plot(ctr, (Cv@datalist[kkk]).T[ctf, 0],
             cmkr, color=ccl, label=labellist[kkk],
             linewidth=lw, markersize=msize)
    ax1.plot(ctr, (Cv@datalist[kkk]).T[ctf, 1:],
             cmkr, color=ccl,
             linewidth=lw, markersize=msize)
    if kkk == 0:
        ctf, ctr = incrmntfilter(ctf)
    else:
        ctf, ctr = incrmntfilter(ctf[1:])

# ax1.plot(T, (Cv@Voptinf).T, '--b')

ax2.set_xlabel('time $t$')
ax2.set_ylabel('$L_{\\infty}$ error of $v(t)$')
ax2.legend(loc='upper right')
ax2.set_title("Approximation error")
# ax2.subplots_adjust(wspace=0.5)
ax1.set_ylabel('$y(t)=C_{v}v(t)$')
ax1.legend(loc='upper right')
ax1.set_title("Time-domain simulation")

if not os.path.exists('Figures'):
    os.makedirs('Figures')

tikzplotlib.save("./Figures/cylinder_wake_time_domain.tex")
plt.show()
fig.savefig("./Figures/cylinder_wake_time_domain.pdf")    

if compute_pod:
    print('POD error: ', norm(Vpod-Vf))

print('Optinf error: ', norm(Voptinf-Vf))
print('Optinf lin error: ', norm(Voptinf_lin-Vf))
print('DMD error: ', norm(Vdmd-Vf))
print('DMDc error: ', norm(Vdmdc-Vf))


