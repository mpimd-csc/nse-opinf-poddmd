from nse_opinf_poddmd.load_data import get_matrices, load_snapshots
# from nse_opinf_poddmd.plotting_tools import plotting_SVD_decay
from nse_opinf_poddmd.optinf_tools import deriv_approx_data, optinf_quad_svd, \
    pod_model, optinf_linear
import nse_opinf_poddmd.optinf_tools as oit
from nse_opinf_poddmd.dmd_tools import dmd_model, dmdc_model, dmdquad_model, \
    sim_dmd, sim_dmdc

import numpy as np
from scipy.linalg import norm
from scipy.linalg import svd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import os
import tikzplotlib


def fab(a, b): return np.concatenate((a, b), axis=0)


# ##########################################################################
# ##### System parameters ##################################################
# ##########################################################################

problem = 'drivencavity'
# Ration between traning and test data
ratio = 0.8
Nprob = 2
nseodedata = False
# nseodedata = True
Re = 500
t0 = 0.
tE = 6  # 4.
# Nts = 2**12
Nts = 2**9
nsnapshots = 2**9

NVdict = {1: 722, 2: 3042, 3: 6962}
NV = NVdict[Nprob]
Re = 500


# Make it come true
plot_results = False
compute_pod = True
compute_pressure = False


if Nprob in [1, 2]:
    tol_lstsq = 1e-7
else:
    tol_lstsq = 1e-7

tol_lstsq_dmdc = 1e-8
###########################################################################
# ##### Loading system data ################################################
###########################################################################

print('Loading data for '+problem + ' problem with NV =', NV, 'and Re =', Re)
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
Vf = V                      # Vf correponds to the test velocity data
Tf = T                      # Tf correponds to the time interval for Tf
V  = Vf[:,:int(len(T)*ratio)] # V correponds to the training velocity data
T  = T[:int(len(T)*ratio)]    # T correponds to the time interval for T



###########################################################################
# ##### Computing reduced basis ############################################
###########################################################################

Uv, Sv, VvT = svd(V)

# plotting decay of singular values
# plotting_SVD_decay(Sv)

# order of reduced models
rv = 30
Uvr = Uv[:, :rv]

print('The reduced basis satisfies the algebraic contrains with an error = ',
      norm(A12.T@Uvr), '\n')

# Reduced basis for pressure
if compute_pressure:
    Up, Sp, VpT = svd(P)

    # plotting decay of singular values
    # plotting_SVD_decay(Sp, 'pressure')

    rp = 30
    Upr = Up[:, :rp]


###########################################################################
# ##### Computing reduced trajectories######################################
###########################################################################

dt = T[1]-T[0]
V_red = Uvr.T@V
N_sims = 0

# Computing reduced derivatives
Vd_red = deriv_approx_data(V_red, dt, N_sims)

if compute_pressure:
    P_red = Upr.T@P

###########################################################################
# ##### Operator inference quadratic model #################################
###########################################################################

print('Computing operator inference model... \n')

if compute_pressure:

    Aoptinf, Hoptinf, Boptinf, Ap, Hp, Bp = optinf_quad_svd(
        V_red, Vd_red, tol_lstsq, P_red)

else:
    Aoptinf, Hoptinf, Boptinf = optinf_quad_svd(V_red, Vd_red, tol_lstsq)


###########################################################################
# ##### Operator inference linear model ####################################
###########################################################################

Aoptinf_lin, Boptinf_lin = optinf_linear(V_red, Vd_red)

###########################################################################
# ##### POD quadratic model ################################################
###########################################################################

if compute_pod:
    print('Computing POD model... \n')
    Apod, Hpod, Bpod, Hpodfunc = pod_model(Uvr, M, A11, H, B1,
                                           ret_hpodfunc=True)

###########################################################################
# ##### DMD  model #########################################################
###########################################################################

print('Computing DMD models... \n')
Admd = dmd_model(Uvr, V, rv)

###########################################################################
# ##### DMD model with control #############################################
###########################################################################

Admdc, Bdmdc = dmdc_model(Uvr, V, rv, tol_lstsq_dmdc)

###########################################################################
# ##### DMD quadratic model with control ###################################
###########################################################################

Admd_quad, Hdmd_quad, Bdmd_quad = dmdquad_model(Uvr, V, rv)

###########################################################################
# ##### Simulatind systems #################################################
###########################################################################

print('Simulating reduced order systems... \n')

# projected initial condition
x0 = Uvr.T@V[:, 0]

# simulating Optinf model
optinf_qm = oit.get_quad_model(A=Aoptinf, H=Hoptinf, B=Boptinf)
xsol_optinf = odeint(optinf_qm, x0, Tf)  # , args=(Aoptinf,Hoptinf,Boptinf))
Voptinf = Uvr @ xsol_optinf.T

# simulatinf OptInf linear model
xsol_optinf_lin = odeint(oit.lin_model, x0, Tf, (Aoptinf_lin, Boptinf_lin))
Voptinf_lin = Uvr @ xsol_optinf_lin.T

# simulating POD model
if compute_pod:
    print('POD...')
    pod_qm = oit.get_quad_model(A=Apod, H=Hpod, B=Bpod, podbase=Uvr)
    pod_qm = oit.get_quad_model(A=Apod, Hfunc=Hpodfunc, B=Bpod)
    xsol_pod = odeint(pod_qm, x0, Tf)  # args=(Apod,Hpod,Bpod))
    Vpod = Uvr @ xsol_pod.T
    print('...done')

# simulating DMD model
Vrdmd = sim_dmd(Admd, x0, len(Tf))
Vdmd = Uvr@Vrdmd

# Simulating DMD model with control
Vrdmdc = sim_dmdc(Admdc, Bdmdc, x0, len(Tf))
Vdmdc = Uvr@Vrdmdc

# Simulating DMD quadratic model with control
# Vrdmd_quad = sim_dmdquad(Admd_quad, Hdmd_quad, Bdmd_quad, x0, len(T))
# Vdmd_quad  = Uvr@Vrdmd_quad


###########################################################################
# ##### Plotting results ###################################################
###########################################################################


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, tight_layout=True)
flngth = 25
tfilter = np.arange(0, len(Tf), flngth)
fskip = 4
trange = np.array(Tf)


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
    ax2.semilogy(ctr, np.max(np.abs(Vf - datalist[kkk]).T, axis=1)[ctf],
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
ax1.axvline(x=T[-1], color='k', linestyle='--')
ax2.axvline(x=T[-1], color='k', linestyle='--')
ax2.set_xlabel('time $t$')
ax2.set_ylabel('$L_{\\infty}$ error of $v(t)$')
#ax2.legend(loc='upper right')
ax2.set_title("Approximation error")
# ax2.subplots_adjust(wspace=0.5)
ax1.set_ylabel('$y(t)=C_{v}v(t)$')
ax1.legend(loc='upper right')
ax1.set_title("Time-domain simulation")


if not os.path.exists('Figures'):
    os.makedirs('Figures')

tikzplotlib.save("Figures/driven_cavity_time_domain_3042.tex")
fig.savefig("Figures/driven_cavity_time_domain_3042.pdf")
plt.show()

if compute_pod:
    print('POD error: ', norm(Vpod-Vf))

print('Optinf error: ', norm(Voptinf-Vf))
print('Optinf lin error: ', norm(Voptinf_lin-Vf))
print('DMD error: ', norm(Vdmd-Vf))
print('DMDc error: ', norm(Vdmdc-Vf))
