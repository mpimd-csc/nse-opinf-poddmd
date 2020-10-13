import numpy as np
from nse_opinf_poddmd.load_data import get_matrices, load_snapshots
from nse_opinf_poddmd.plotting_tools import print_nparray_tex

cwNVdict = {1: 5812, 2: 9356, 3: 19468}
dvNVdict = {1: 722, 2: 3042, 3: 6962}


def comp_approx_err(problem=None, Re=None,
                    tE=None, nsnapshots=None,
                    Ncheck=None, Ntscheck=None,
                    CVref=None,
                    CVivref=0., CVevref=0., getref=False):

    NVdict = cwNVdict if problem == 'cylinderwake' else dvNVdict
    NVcheck = NVdict[Ncheck]
    DT = tE/nsnapshots

    # getting system matrices
    M, _, _, _, _, _, Cv, _ = get_matrices(problem, NVcheck)

    # loading snapshots
    # V, Vd, MVd, P, T = load_snapshots(1, NV, problem, Re,
    #                                   False, False, odeint=nseodedata)

    V, _, _, _, _ = load_snapshots(N=Ncheck, problem=problem,
                                   Re=Re, tE=tE, Nts=Ntscheck,
                                   nsnapshots=nsnapshots,
                                   odesolve=getref)
    vzero = V[:, 0:1]
    sumdnrm = np.sqrt(vzero.T.dot(M*vzero)).flatten()[0]
    for k in range(nsnapshots):
        vk = V[:, k+1:k+2]
        sumdnrm += vk.T.dot(M*vk).flatten()[0]

    if getref:
        # return (checkCViv, checkCVev)
        return Cv.dot(V), DT*sumdnrm

    if CVref is not None:
        CVdiff = CVref - Cv.dot(V)
        inierr = np.linalg.norm(CVdiff[:, 0])
        sumderr = inierr
        for k in range(nsnapshots):
            sumderr += np.linalg.norm(CVdiff[:, k+1])
        return (inierr, DT*sumderr, DT*sumdnrm)

    else:
        checkCViv = Cv.dot(V[:, 0])
        checkCVev = Cv.dot(V[:, -1])
        return (np.linalg.norm(checkCViv-CVivref),
                np.linalg.norm(checkCVev-CVevref))


if __name__ == '__main__':

    Nref = 3
    Nlist = [1, 2, 3]
    problem = 'drivencavity'
    problem = 'cylinderwake'
    if problem == 'cylinderwake':
        Re = 60
        tE = 2.  # 0.4
        nsnapshots = 2**9
        Ntslist = [2**k for k in [9, 10, 11]]
    elif problem == 'drivencavity':
        Re = 500
        t0 = 0.
        tE = 6  # 4.
        # Nts = 2**12
        Ntslist = [2**k for k in [8, 9, 10]]
        nsnapshots = 2**8

    CVref, vnrmref = comp_approx_err(problem=problem, Re=Re,
                                     tE=tE, nsnapshots=nsnapshots,
                                     Ncheck=Nref, Ntscheck=Ntslist[-1],
                                     CVivref=0., CVevref=0., getref=True)

    print(vnrmref)

    everrlist = []
    iverrlist = []
    nrmlist = []
    print('Nts: ', Ntslist)
    for Ncheck in Nlist:
        locerrlist = []
        locnrmlist = []
        for Ntscheck in Ntslist:
            ive, eve, nrmv = comp_approx_err(problem=problem, Re=Re,
                                             tE=tE, nsnapshots=nsnapshots,
                                             Ncheck=Ncheck, Ntscheck=Ntscheck,
                                             CVref=CVref)
            locerrlist.append(eve)
            locnrmlist.append(nrmv-vnrmref)
        iverrlist.append(ive)
        everrlist.append(np.array(locerrlist))
        nrmlist.append(np.array(locnrmlist))
        print('N={0}'.format(Ncheck), locerrlist)
        print('N={0}'.format(Ncheck), locnrmlist)
    everrs = np.hstack(everrlist)
    print(iverrlist)
    print(everrs)
    print(nrmlist)

print_nparray_tex(nrmlist)
