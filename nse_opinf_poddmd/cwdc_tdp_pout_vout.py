import numpy as np
import scipy.io
import scipy.sparse as sps
import scipy.sparse.linalg as spsla
from scipy.integrate import solve_ivp
# from scipy.integrate import odeint
# import conv_tensor_utils as ctu
import nse_opinf_poddmd.visualization_utils as vu
import sys
import getopt
import os
import json

debug = True
debug = False


def comp_snapshots(N=1, NV=None, problem='drivencavity',
                   t0=0.,
                   Re=None, tE=None, Nts=None, nsnapshots=None,
                   verbose=False, returngradp=False, odesolve=False):

    def savedmatsstr(
        NV): return 'data/' + problem + '__mats_NV{1}_Re{0}.mat'.format(1, NV)

    def visujsonstr(
        NV): return 'data/visualization_' + problem + '_NV{0}.jsn'.format(NV)

    if problem == 'cylinderwake':
        NVdict = {1: 5812, 2: 9356, 3: 19468}
    else:
        NVdict = {1: 722, 2: 3042, 3: 6962}

    veldict = {}
    prsdict = {}
    gradprsdict = {}
    momrhsdict = {}
    minvmomrhsdict = {}
    contirhsdict = {}

    # further parameters
    NV = NVdict[N]
    if odesolve:
        DT = None
        trange = None
    else:
        DT = (tE-t0)/Nts
        trange = np.linspace(t0, tE, Nts+1)
    snshtrange = np.linspace(t0, tE, nsnapshots+1)

    # parameters for results, directories
    rdir = 'results/'
    vfileprfx = 'v_'+problem + \
        '_NV{0}_Re{1}_tE{2}_Nts{3}'.format(NV, Re, tE, Nts)
    pfileprfx = 'p_'+problem + \
        '_NV{0}_Re{1}_tE{2}_Nts{3}'.format(NV, Re, tE, Nts)

    poutlist = []
    voutlist = []

    def vfile(t): return rdir + vfileprfx + '__t{0}.vtu'.format(t)

    def pfile(t): return rdir + pfileprfx + '__t{0}.vtu'.format(t)

    vfilelist = [vfile(snshtrange[0])]
    pfilelist = [pfile(snshtrange[0])]
    # ptikzfile = 'tikz/p_nsequadtens-N{0}-tE{1}-Nts{2}'.format(N, tE, Nts)
    # vtikzfile = 'tikz/v_nsequadtens-N{0}-tE{1}-Nts{2}'.format(N, tE, Nts)

    if odesolve:
        solverstr = '_odeint_'
        Nts = 'dna'
    else:
        solverstr = ''
    datastr = 'snapshots_' + problem + solverstr + \
        '_Re{1}_NV{0}_tE{2}_Nts{3}_nsnaps{4}.json'.format(NV, Re, tE,
                                                          Nts, nsnapshots)
    savesnapshtstrv = 'results/vel_' + datastr
    savesnapshtstrp = 'results/prs_' + datastr
    savesnapshtstrgradp = 'results/gradprs_' + datastr
    savesnapshtstrmomrhs = 'results/momrhs_' + datastr
    savesnapshtstrcontirhs = 'results/contirhs_' + datastr
    savesnapshtstrminvmomrhs = 'results/minv_momrhs_' + datastr

    # create dir if not exists
    if not os.path.exists('results'):
        os.makedirs('results')

    if not os.path.exists('tikz'):
        os.makedirs('tikz')

    # print reynolds number, discretization lvl, and other params
    print(('Problem      = {0}'.format(problem)))
    print(('Re           = {0}'.format(Re)))
    print(('NV           = {0}'.format(NV)))
    print(('t0           = {0}'.format(t0)))
    print(('tE           = {0}'.format(tE)))
    print(('Nts          = {0}'.format(Nts)))
    print(('DT           = {0}'.format(DT)))
    print('\n')

    # load the coefficients matrices
    mats = scipy.io.loadmat(savedmatsstr(NV))
    M = mats['M']
    A = 1./Re*mats['A'] + mats['L1'] + mats['L2']
    J = mats['J']
    hmat = mats['H']
    fv = mats['fv'] + 1./Re*mats['fv_diff'] + mats['fv_conv']
    fp = mats['fp'] + mats['fp_div']
    pcmat = mats['Cp']
    vcmat = mats['Cv']

    if problem == 'drivencavity':
        # Fix the p
        J = J[:-1, :]
        fp = fp[:-1, :]

    NV, NP = fv.shape[0], fp.shape[0]

    # factorization of system matrix
    print('computing LU once...')
    if odesolve or debug:
        sadptmat = sps.vstack([sps.hstack([M, J.T]),
                               sps.hstack([J, sps.csc_matrix((NP, NP))])]
                              ).tocsc()
        invsadptmat = spsla.factorized(sadptmat)
        rhsvperp = np.vstack([0*fv, fp])
        vperp = (invsadptmat(rhsvperp.flatten())[:NV]).reshape((NV, 1))

        zfpvec = 0*fp.flatten()

        def prjctvelvec(vvec):
            prjtrhs = np.r_[vvec, zfpvec]
            return invsadptmat(prjtrhs)[:NV]

        # def nseoderhs(vzero, t):
        def nseoderhs(t, vzero):
            vzvp = vzero.reshape((NV, 1)) + vperp
            # momrhs = -A.dot(vzvp) - ctu.eva_quadterm(hmat, vzvp) + fv
            momrhs = -A.dot(vzvp) - hmat*np.kron(vzvp, vzvp) + fv
            if debug:
                # print(t, momrhs.shape, np.linalg.norm(momrhs))
                # , momrhs.shape, np.linalg.norm(momrhs))
                print('time is: ', t)
            return prjctvelvec(momrhs.flatten())
            # minvmomrhs = mmati(momrhs.flatten())
            # return prjctvelvec(minvmomrhs)

        def getpfromv(cvel):
            cvel = cvel.reshape((NV, 1))
            convterm = (hmat*np.kron(cvel, cvel)).flatten()
            pperhs = np.r_[fv.flatten()-(A*cvel).flatten()-convterm, zfpvec]
            return -invsadptmat(pperhs)[NV:]

    if not odesolve:
        sysmat = sps.vstack([sps.hstack([M+DT*A, -DT*J.T]),
                             sps.hstack([J, sps.csc_matrix((NP, NP))])]
                            ).tocsc()
        sysmati = spsla.factorized(sysmat)
    mmati = spsla.factorized(M)

    # compute stokes solution as initial value
    print('computing Stokes solution to be used as initial value...')
    fvstks = mats['fv'] + 1./Re*mats['fv_diff']
    Astks = 1./Re*mats['A']
    stksrhs = np.vstack([fvstks, fp])
    stksmat = sps.vstack([sps.hstack([Astks, -J.T]),
                          sps.hstack([J, sps.csc_matrix((NP, NP))])]).tocsc()
    stksvp = spsla.spsolve(stksmat, stksrhs).reshape((NV+NP, 1))
    stksv = stksvp[:NV].reshape((NV, 1))
    stksp = stksvp[NV:].reshape((NP, 1))
    if problem == 'drivencavity':
        pforplt = (np.r_[stksp.flatten(), 0]).reshape((NP+1, 1))
    else:
        pforplt = stksp

    # Preparing for the output
    vu.writevp_paraview(velvec=stksv, pvec=pforplt, vfile=vfile(
        snshtrange[0]), pfile=pfile(snshtrange[0]), strtojson=visujsonstr(NV))
    prsdict.update({snshtrange[0]: stksp.tolist()})
    veldict.update({snshtrange[0]: stksv.tolist()})
    gradprsdict.update({snshtrange[0]: (J.T.dot(stksp)).tolist()})

    # time stepping
    print('doing the time loop...')
    old_v = stksv
    # Adding pertubation to initial condition
    # pert = np.random.randn(NV, 1)
    # pert_final = pert - J.T@np.linalg.solve(J@J.T.todense(), J@pert)

    # old_v = stksv+1e-1*pert_final
    # old_v = stksv+1e-1*pert_final
    # old_v = pert_final/np.linalg.norm(pert_final)*np.linalg.norm(stksv)
    # cfv = fv - ctu.eva_quadterm(hmat, old_v)
    cfv = fv - hmat*np.kron(old_v, old_v)
    mvdot = -A.dot(old_v) + J.T.dot(stksp) + cfv
    vdot = mmati(mvdot.flatten())

    momrhsdict.update({snshtrange[0]: mvdot.tolist()})
    minvmomrhsdict.update({snshtrange[0]: vdot.tolist()})
    contirhsdict.update({snshtrange[0]: fp.tolist()})

    if debug:
        # nseodeexpts = old_v.flatten() + DT*nseoderhs(old_v, 0)
        nseodeexpts = old_v.flatten() + DT*nseoderhs(0, old_v)
        print(np.linalg.norm(J*nseoderhs(0, old_v)))
        print(np.linalg.norm(old_v.flatten() - prjctvelvec(M*old_v.flatten())))

    if odesolve:
        prjdvini = prjctvelvec(M*old_v.flatten())
        # nseodesol = odeint(nseoderhs, prjdvini, trange)
        nseodesol = solve_ivp(nseoderhs, (t0, tE), prjdvini,
                              t_eval=snshtrange, method='RK23').y

        for tk, t in enumerate(snshtrange):
            cvel = nseodesol[:, tk] + vperp.flatten()
            cprs = getpfromv(cvel)
            veldict.update({t: cvel.tolist()})
            prsdict.update({t: cprs.tolist()})
            if debug or verbose:
                print(('snapshot {0:4d}/{1}, t={2:f}, |v|={3:e}'.
                       format(tk, nsnapshots, t, np.linalg.norm(cvel))))
                print(('snapshot {0:4d}/{1}, t={2:f}, |p|={3:e}'.
                       format(tk, nsnapshots, t, np.linalg.norm(cprs))))

    else:
        for k, t in enumerate(trange[1:]):
            k = k+1
            crhsv = M*old_v + DT*cfv
            crhs = np.vstack([crhsv, fp])
            vp_new = np.atleast_2d(sysmati(crhs.flatten())).T
            old_old_v = np.copy(old_v)
            old_v = vp_new[:NV]
            p = vp_new[NV:]
            res = (M+DT*A).dot(old_v) - DT*J.T*p - M*old_old_v - DT*cfv
            if debug:
                print(t, np.linalg.norm(old_v.flatten()-nseodeexpts -
                                        vperp.flatten()))
                nseodeexpts = old_v.flatten() + DT*nseoderhs(old_v, t)
                print(np.linalg.norm(res))

            cfv_old = np.copy(cfv)
            # cfv = fv - ctu.eva_quadterm(hmat, old_v)
            cfv = fv - hmat*np.kron(old_v, old_v)

            if problem == 'drivencavity':
                pforplt = (np.r_[p.flatten(), 0]).reshape((NP+1, 1))
            else:
                pforplt = p

            poutlist.append((pcmat*pforplt)[0][0])
            voutlist.append((vcmat*old_v).flatten())
            if np.mod(k, round(Nts/nsnapshots)) == 0:
                if debug or verbose:
                    print(('timestep {0:4d}/{1}, t={2:f}, |v|={3:e}'.
                           format(k, Nts, t, np.linalg.norm(old_v))))
                    print(('timestep {0:4d}/{1}, t={2:f}, |p|={3:e}'.
                           format(k, Nts, t, np.linalg.norm(p))))
                vu.writevp_paraview(velvec=old_v, pvec=pforplt, vfile=vfile(
                    t), pfile=pfile(t), strtojson=visujsonstr(NV))
                vfilelist.append(vfile(t))
                pfilelist.append(pfile(t))
                veldict.update({t: old_v.tolist()})
                prsdict.update({t: p.tolist()})
                gradprsdict.update({t: (J.T.dot(p)).tolist()})
                mvdot = -A.dot(old_v) + J.T.dot(p) + cfv_old
                vdot = mmati(mvdot.flatten())
                # diffinderiv = 1/DT*(old_v - old_old_v).flatten() - vdot
                # res = 1/DT*M.dot(old_v-old_old_v)+A.dot(old_v)-J.T*p-cfv_old
                # print(np.linalg.norm(res))
                # print('diff in deriv: ', np.linalg.norm(diffinderiv))
                momrhsdict.update({t: mvdot.tolist()})
                minvmomrhsdict.update({t: vdot.tolist()})
                contirhsdict.update({t: fp.tolist()})

    # save collection to pvd file
    # vu.collect_vtu_files(vfilelist, vfileprfx+'.pvd')
    # vu.collect_vtu_files(pfilelist, pfileprfx+'.pvd')

    # write to tikz file
    # vu.plot_prs_outp(outsig=poutlist, tmesh=trange, tikzfile=ptikzfile)
    # vu.plot_prs_outp(outsig=voutlist, tmesh=trange, tikzfile=vtikzfile)
    # firstvel = np.array(veldict[snshtrange[0]]).flatten()
    # print(np.linalg.norm(firstvel - stksv.flatten()))

    with open(savesnapshtstrv, 'w') as outfile:
        json.dump(veldict, outfile)
    with open(savesnapshtstrp, 'w') as outfile:
        json.dump(prsdict, outfile)
    with open(savesnapshtstrgradp, 'w') as outfile:
        json.dump(gradprsdict, outfile)
    with open(savesnapshtstrmomrhs, 'w') as outfile:
        json.dump(momrhsdict, outfile)
    with open(savesnapshtstrminvmomrhs, 'w') as outfile:
        json.dump(minvmomrhsdict, outfile)
    with open(savesnapshtstrcontirhs, 'w') as outfile:
        json.dump(contirhsdict, outfile)


if __name__ == '__main__':
    problem = 'cylinderwake'
    problem = 'drivencavity'
    odesolve = True
    odesolve = False
    # setup parameters
    if problem == 'cylinderwake':
        N = 1
        Re = 60  # 40
        t0 = 0.
        tE = 2  # 0.4
        Nts = 2**10
        nsnapshots = 2**8
        # tE = 1.  # 0.4
        # Nts = 2**8
        # nsnapshots = 2**4
    elif problem == 'drivencavity':
        N = 2
        Re = 500
        t0 = 0.
        tE = 3  # 4.
        # Nts = 2**12
        Nts = 2**9
        nsnapshots = 2**9

    # parameters for time stepping

    # get command line input and overwrite standard paramters if necessary
    options, rest = getopt.getopt(
        sys.argv[1:], '', ['N=', 'Re=', 'Picardsteps=', 't0=', 'tE=', 'Nts='])
    for opt, arg in options:
        if opt == '--N':
            N = int(arg)
        elif opt == '--Re':
            Re = int(arg)
        elif opt == '--t0':
            t0 = float(arg)
        elif opt == '--tE':
            tE = float(arg)
        elif opt == '--Nts':
            Nts = int(arg)
    comp_snapshots(N=N, problem=problem, Re=Re, tE=tE, Nts=Nts,
                   nsnapshots=nsnapshots, verbose=False, returngradp=False,
                   odesolve=odesolve)
