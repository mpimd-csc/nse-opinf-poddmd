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

def fab(a, b): return np.concatenate((a, b), axis=0)


###########################################################################
###### System parameters ##################################################
###########################################################################

problem = 'drivencavity'
Nprob = 2
nseodedata = False
#nseodedata = True

if problem == 'cylinderwake':
    NVdict = {1: 5812, 2: 9356, 3: 19468}
    NV = NVdict[Nprob]
    Re = 40
    t0 = 0.
    tE = 2  # 0.4
    Nts = 2**9
    nsnapshots = 2**9
else:
    NVdict = {1: 722, 2: 3042, 3: 6962}
    NV = NVdict[Nprob]
    Re = 500
    t0 = 0.
    tE = 6
    Nts = 2**9
    nsnapshots = 2**9

# Make it come true
plot_results        = True
compute_pod         = False
compute_pressure    = True


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

#P[:,0] = P[:,1];
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



if plot_results:
    print('Plotting results... \n')
    # plotting observed trajectories
    if compute_pod:
        plotting_obv_vel(Vf, Voptinf, Vpod, Vdmdc, Vdmd_quad, T, Cv)
        plotting_abs_error(Vf, Vpod, T, 'POD')
    
    plt.figure()
    plt.plot(T, (Cv@V).T, 'b', label='original')
    plt.plot(T, (Cv@Voptinf).T, '--g', label='optinf')
    plt.plot(T, (Cv@Voptinf_lin).T, '--m', label='optinf_lin')
    plt.plot(T, (Cv@Vdmdc).T,'--y', label = 'DMDc')
    

    #plt.legend(loc='best')
    plt.xlabel('t')
    # ploting absolute erros
    plotting_abs_error(Vf, Voptinf, T, 'Optinf' )
    plotting_abs_error(Vf, Vdmd, T, 'DMD')
    plotting_abs_error(Vf, Vdmdc, T, 'DMDc')
    

if compute_pod:
    print('POD error: ', norm(Vpod-Vf))

print('Optinf error: ', norm(Voptinf-Vf))
print('Optinf lin error: ', norm(Voptinf_lin-Vf))
print('DMD error: ', norm(Vdmd-Vf))
print('DMDc error: ', norm(Vdmdc-Vf))
#print('DMDquad error: ', norm(Vdmd_quad-Vf))


if compute_pressure:
    V_kron_red  = np.array([np.kron(V_red[:,i],V_red[:,i]) for i in range(V_red.shape[1])]).T
    Proptinf = Ap@V_red +Hp@V_kron_red + Bp.reshape(-1,1)
    Poptinf = Upr@Proptinf

    fig2 = plt.figure()

    ax = plt.subplot(131)
    #ax.plot(T,(P).T)
    ax.plot(T,(Cp[0,1:]@P).T)
    plt.xlabel('time (sec)')
    plt.ylabel('pressure')
    plt.title('FOM')
    ax = plt.subplot(132)
    #ax.plot(T,(Poptinf).T)
    ax.plot(T,(Cp[0,1:]@Poptinf).T)
    plt.xlabel('time (sec)')
    plt.title('OpInf')
    ax = plt.subplot(133)
    ax.plot(T,np.mean(np.abs(P-Poptinf).T,axis=1))
    plt.xlabel('time (sec)')
    plt.title('Error')
    plt.subplots_adjust(wspace = 0.3)
    
    
    print('\nOptinf pressure error: ', norm(Poptinf-P))
    #tikzplotlib.save("./Figures/driven_cavity_pressure_3042.tex")
    #plt.show()
    #fig2.savefig("./Figures/driven_cavity_pressure_3042.pdf")   
# Take 2**9 snapshots, Tend = 6, rv = 30 
#
#
plt.show()
