import numpy as np
from nse_opinf_poddmd.load_data import get_matrices, load_snapshots
from nse_opinf_poddmd.plotting_tools import plotting_SVD_decay, plotting_obv_vel, plotting_abs_error
from nse_opinf_poddmd.optinf_tools import deriv_approx_data, optinf_quad_svd, pod_model
import nse_opinf_poddmd.optinf_tools as oit
from nse_opinf_poddmd.dmd_tools import dmd_model, dmdc_model, dmdquad_model, sim_dmd, \
    sim_dmdc, sim_dmdquad
from scipy.linalg import norm
from scipy.linalg import svd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.linalg import solve
import scipy as sp
import tikzplotlib, os

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

Uv_divfree, Sv_divfree, VvT_divfree = svd(V_divfree)

err_pod_alg = []
# order of reduced models
for rv in range(1,150):

    Uvr = Uv_divfree[:,:rv]
    print('The reduced basis satisfies the algebraic contrains with an error = '
          , norm(A12.T@Uvr))
    print('\n')
    err_pod_alg.append(norm(A12.T@Uvr)/norm(Uvr))
    
range_rv = range(1,150)
fig = plt.figure()

# Decay of POD singular values
ax = plt.subplot(121)
ax.semilogy(Sv[:150], 'g-o',label = 'vel.')
ax.semilogy(Sv_divfree[:150],'b--',label = 'div.-free vel.')

# Set common labels
ax.set_xlabel('k')
ax.set_ylabel('singular values')
plt.legend()
plt.legend()
ax.set_title("Singular values decay")

# Algebraic conditions error
ax = plt.subplot(122)
ax.semilogy(range_rv,err_pod_alg,label='POD basis')
plt.xlabel('Reduced order $r$')
plt.ylabel('$\|\mathbf{A}_{12}^T \mathbf{V}_v\|_F/\|\mathbf{V}_v\|_F$')
plt.subplots_adjust(wspace = 0.5)
ax.legend()
ax.set_title("alg. cond. error")
plt.subplots_adjust(wspace = 0.5)
if not os.path.exists('Figures'):
    os.makedirs('Figures')
    
tikzplotlib.save("./Figures/cylinder_wake_pod_decay_and_alg_cond.tex")
plt.show()
fig.savefig("./Figures/cylinder_wake_pod_decay_and_alg_cond.pdf")
