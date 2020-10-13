import numpy as np
from nse_opinf_poddmd.load_data import get_matrices, load_snapshots
from nse_opinf_poddmd.plotting_tools import plotting_SVD_decay, plotting_obv_vel, plotting_abs_error
from nse_opinf_poddmd.optinf_tools import deriv_approx_data, optinf_quad_svd, pod_model
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


if plot_results:
    print('Plotting results... \n')
    # plotting observed trajectories
    if compute_pod:
        plotting_obv_vel(Vf, Voptinf, Vpod, Vdmdc, Vdmd_quad, T, Cv)
        plotting_abs_error(Vf, Vpod, T, 'POD')
    
    
    # ploting absolute erros
    plotting_abs_error(Vf, Voptinf, T, 'Optinf' )
    plotting_abs_error(Vf, Vdmd, T, 'DMD')
    plotting_abs_error(Vf, Vdmdc, T, 'DMDc')
    #plotting_abs_error(Vf, Vpod, T, 'DMDc')

    import nse_opinf_poddmd.visualization_utils as vu

    def visujsonstr(
        NV): return 'data/visualization_' + problem + '_NV{0}.jsn'.format(NV)
    # vu.writevp_paraview(velvec=Voptinf[:, -2:-1],
    # vu.writevp_paraview(velvec=Vf[:, -2:-1],
    vu.writevp_paraview(velvec=Voptinf[:, -2:-1]-Vf[:, -2:-1],
                        ignore_bcs=True,
                        vfile='Figures/cyl-diff-opinf.vtu', strtojson=visujsonstr(NV))
    vu.writevp_paraview(velvec=Vdmdc[:, -2:-1]-Vf[:, -2:-1],
                        ignore_bcs=True,
                        vfile='Figures/cyl-diff-dmdc.vtu', strtojson=visujsonstr(NV))

if compute_pod:
    print('POD error: ', norm(Vpod-Vf))

print('Optinf error: ', norm(Voptinf-Vf))
print('DMD error: ', norm(Vdmd-Vf))
print('DMDc error: ', norm(Vdmdc-Vf))
#print('DMDquad error: ', norm(Vdmd_quad-Vf))


# Take 2**9 snapshots, Tend = 6, rv = 30 
#
#
plt.show()
