"""Loading examples matrices and snaphots data"""

__all__ = [
            "get_matrices",
            "load_snapshots"
          ]

import scipy.io
import numpy as np
import json

from nse_opinf_poddmd.cwdc_tdp_pout_vout import comp_snapshots


def savedmatsstr(problem, NV):
    return 'data/' + problem + \
        '__mats_NV{1}_Re{0}.mat'.format(1, NV)

## Getting matrcies function
################################

def get_matrices(problem, NV):
    # function returning matrices defining the example differential eq. 
    #
    # M*dx = A11*x + A12*p H*kron(x,x) + B1
    #    0 = A12.T*x + B2  
    #
    if problem == 'cylinderwake':
        Re = 60
        
   
    elif problem == 'drivencavity':
        Re = 500
    
    # geting matrix from .mat    
    mats = scipy.io.loadmat(savedmatsstr(problem, NV))
   
    # Mass matrix
    M   = mats['M']
    
    # matrix A11
    A   = 1./Re*mats['A'] + mats['L1'] + mats['L2']
    A11 = -A 
    
    # matrix A12
    J   = mats['J']
    A12 = J.T
    
    # quadratic part H
    H = mats['H']
    H = -H
    
    # Input matrix B1
    fv = mats['fv'] + 1./Re*mats['fv_diff'] + mats['fv_conv']
    B1 = fv
    
    # Input matrix B2
    fp = mats['fp'] + mats['fp_div']
    B2 = fp
    
    # Observation matrices Cv and Cp
    Cp = mats['Cp']
    Cv = mats['Cv']
    
    return M, A11, A12, H, B1, B2, Cv, Cp

## Loading snapshots function
################################


def load_snapshots(N=1, NV=None, problem='drivencavity',
                   Re=500, tE=None, Nts=None, nsnapshots=None,
                   verbose=False, returngradp=False, odesolve=False):
    if odesolve:
        solverstr = '_odeint_'
        
    else:
        solverstr = ''

    if odesolve:
        Nts = 'dna'

    # hard coded paths and dictionary for data
    if problem == 'cylinderwake':
        NVdict = {1: 5812, 2: 9356, 3: 19468}
    elif problem == 'drivencavity':
        NVdict = {1: 722, 2: 3042, 3: 6962}
    if NV is None:
        NV = NVdict[N]
    datastr = 'snapshots_' + problem + solverstr + \
        '_Re{1}_NV{0}.json'.format(NV, Re)
    datastr = 'snapshots_' + problem + solverstr + \
        '_Re{1}_NV{0}_tE{2}_Nts{3}_nsnaps{4}.json'.format(NV, Re, tE,
                                                          Nts, nsnapshots)
    savesnapshtstrv = 'results/vel_' + datastr
    savesnapshtstrp = 'results/prs_' + datastr
    savesnapshtstrmomrhs = 'results/momrhs_' + datastr
    savesnapshtstrcontirhs = 'results/contirhs_' + datastr
    savesnapshtstrminvmomrhs = 'results/minv_momrhs_' + datastr
    savesnapshtstrgradp = 'results/gradprs_' + datastr

    try:
        with open(savesnapshtstrv) as vdatafile:
            veldict = json.load(vdatafile)
        with open(savesnapshtstrp) as pdatafile:
            prsdict = json.load(pdatafile)
        with open(savesnapshtstrgradp) as jtpdatafile:
            jtprsdict = json.load(jtpdatafile)
        with open(savesnapshtstrmomrhs) as momrhsdatafile:
            momrhsdict = json.load(momrhsdatafile)
        with open(savesnapshtstrminvmomrhs) as minvmomrhsdatafile:
            minvmomrhsdict = json.load(minvmomrhsdatafile)
        with open(savesnapshtstrcontirhs) as contirhsdatafile:
            contirhsdict = json.load(contirhsdatafile)
    except IOError:
        print('no data -- gonna compute it')
        comp_snapshots(N=N, problem=problem,
                       Re=Re, tE=tE, Nts=Nts, nsnapshots=nsnapshots,
                       odesolve=odesolve)
        with open(savesnapshtstrv) as vdatafile:
            veldict = json.load(vdatafile)
        with open(savesnapshtstrp) as pdatafile:
            prsdict = json.load(pdatafile)
        with open(savesnapshtstrgradp) as jtpdatafile:
            jtprsdict = json.load(jtpdatafile)
        with open(savesnapshtstrmomrhs) as momrhsdatafile:
            momrhsdict = json.load(momrhsdatafile)
        with open(savesnapshtstrminvmomrhs) as minvmomrhsdatafile:
            minvmomrhsdict = json.load(minvmomrhsdatafile)
        with open(savesnapshtstrcontirhs) as contirhsdatafile:
            contirhsdict = json.load(contirhsdatafile)

    # Assining matrices to incorporate the snapshots
    V = np.empty((NV, 0), float)
    MVd = np.empty((NV, 0), float)
    Vd = np.empty((NV, 0), float)
    Plist = []
    Vlist = []
    JTP = np.empty((NV, 0), float)

    times = veldict.keys()
    for time in times:
        # Collecting velocity snapshots in a matrix
        varray = np.array(veldict[time])
        NV = varray.size
        Vlist.append(varray.reshape((NV, 1)))
        # V = np.append(V, varray, axis=1)
        # Collecting pressure snapshots in a matrix
        parray = np.array(prsdict[time])
        NP = parray.size
        Plist.append(parray.reshape((NP, 1)))  # , parray, axis=1)
        # Collecting grad-pressure snapshots in a matrix
        try:
            jtparray = np.array(jtprsdict[time])
            JTP = np.append(JTP, jtparray, axis=1)
            # Collecting moment snapshots in a matrix
            momrhs = np.array(momrhsdict[time])
            MVd = np.append(MVd, momrhs, axis=1)
            # Collecting accelaration snapshots in a matrix
            minvmomrhs = np.array([minvmomrhsdict[time]]).T
            Vd = np.append(Vd, minvmomrhs, axis=1)
            # Continuity equation
            contirhs = np.array(contirhsdict[time])
            if verbose:
                t = np.float(time)
                print('time: {0:.4f} -- |v|: {1:.2e}'.
                      format(t, np.linalg.norm(varray)))
                print('time: {0:.4f} -- |p|: {1:.2e}'.
                      format(t, np.linalg.norm(parray)))
                print('time: {0:.4f} -- |rhs(momentum eq)|: {1:.2e}'.
                      format(t, np.linalg.norm(momrhs)))
                print('time: {0:.4f} -- |M^(-1)*rhs(momentum eq)|: {1:.2e}'.
                      format(t, np.linalg.norm(minvmomrhs)))
                print('time: {0:.4f} -- |rhs(continty eq)|: {1:.2e}'.
                      format(t, np.linalg.norm(contirhs)))
        except KeyError:
            pass
            # print("no snaps for `v'` et al.")

    T = np.array(list(times))
    T = list(map(float, T))
    P = np.hstack(Plist)
    V = np.hstack(Vlist)

    if returngradp:
        return V, Vd, MVd, P, JTP, T
    else:
        return V, Vd, MVd, P, T


if __name__ == '__main__':
    problem = 'drivencavity'
    # problem = 'cylinderwake'

    if problem == 'drivencavity':
        Nprob = 2
        NVdict = {1: 722, 2: 3042, 3: 6962}
        NV = NVdict[Nprob]
        Re = 500
        tE = 3  # 4.
        Nts = 2**9
        nsnapshots = 2**9

    if problem == 'cylinderwake':
        Nprob = 1
        NVdict = {1: 5812, 2: 9356, 3: 19468}
        NV = NVdict[Nprob]
        Re = 60
        tE = 2  # 0.4
        Nts = 2**10
        nsnapshots = 2**8

    # getting system matrices
    M, A11, A12, H, B1, B2, Cv, Cp = get_matrices(problem, NV)

    # loading snapshots
    Vodeint, _, _, _z, Todeint =\
        load_snapshots(N=Nprob, problem=problem, Re=Re, tE=tE, Nts=Nts,
                       nsnapshots=nsnapshots, odesolve=True)
    V, _, _, _, T =\
        load_snapshots(N=Nprob, problem=problem, Re=Re, tE=tE, Nts=Nts,
                       nsnapshots=nsnapshots, odesolve=False)
    import matplotlib.pyplot as plt
    plt.figure(1)
    try:
        plt.plot(T, (Cv*V).T-(Cv*Vodeint).T)
    except ValueError:
        plt.plot(T, (Cv*V).T)
        plt.plot(Todeint, (Cv*Vodeint).T)
    plt.figure(2)
    plt.plot(Todeint, (Cv*Vodeint).T)
    # plt.plot(Todeint, (Cv*Vodeint).T)
    plt.show()
