
"""
Script generating the curve showing how much POD basis satisfies the algebraic conditions

"""

from nse_opinf_poddmd.load_data import get_matrices, load_snapshots

import numpy as np
from scipy.linalg import norm
from scipy.linalg import svd
import matplotlib.pyplot as plt
import tikzplotlib
import os



def fab(a, b): return np.concatenate((a, b), axis=0)


###########################################################################
###### System parameters ##################################################
###########################################################################

problem = 'drivencavity'
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
plot_results        = True
compute_pod         = False
compute_pressure    = True


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

# Training and test data
Vf = V                      # Vf correponds to the test velocity data
Tf = T                      # Tf correponds to the time interval for Tf
V  = Vf[:,:int(len(T)*ratio)] # V correponds to the training velocity data
T  = T[:int(len(T)*ratio)]    # T correponds to the time interval for T


###########################################################################
###### Computing reduced basis ############################################
###########################################################################

Uv, Sv, VvT = svd(V)
          

err_pod_alg = []
# order of reduced models
for rv in range(1,150):

    Uvr = Uv[:,:rv]
    print('The reduced basis satisfies the algebraic contrains with an error = '
          , norm(A12.T@Uvr))
    print('\n')
    err_pod_alg.append(norm(A12.T@Uvr)/norm(Uvr))

range_rv = range(1,150)
fig = plt.figure()

# Decay of POD singular values
ax = plt.subplot(121)
ax.semilogy(Sv[:150],label = 'velocity')
# Set common labels
ax.set_xlabel('k')
ax.set_ylabel('singular values')
plt.legend()
ax.set_title("POD singular velues decay")

# Algebraic conditions error
ax = plt.subplot(122)
ax.semilogy(range_rv,err_pod_alg,label='POD basis')
plt.xlabel('Reduced order $r$')
plt.ylabel('$\|\mathbf{A}_{12}^T \mathbf{V}_v\|_F/\|\mathbf{V}_v\|_F$')
plt.subplots_adjust(wspace = 0.5)
ax.legend()
ax.set_title("POD basis error for the alg. cond.")

if not os.path.exists('Figures'):
    os.makedirs('Figures')
    
tikzplotlib.save("Figures/driven_cavity_pod_decay_and_alg_cond_3042.tex")
plt.show()
fig.savefig("Figures/driven_cavity_pod_decay_and_alg_cond_3042.pdf")
