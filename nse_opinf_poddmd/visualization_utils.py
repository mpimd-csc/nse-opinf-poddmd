from __future__ import print_function 
import numpy as np
import json
import itertools
from io import open


def load_json_dicts(StrToJs):
    fjs = open(StrToJs)
    JsDict = json.load(fjs)
    return JsDict


def plot_prs_outp(str_to_json=None, tmeshkey='tmesh', sigkey='outsig',
                  outsig=None, tmesh=None, fignum=333, reference=None,
                  compress=5, tikzfile=None, tikzonly=False):
    import matplotlib.pyplot as plt

    if str_to_json is not None:
        jsdict = load_json_dicts(str_to_json)
        tmesh = jsdict[tmeshkey]
        outsig = jsdict[sigkey]
    else:
        str_to_json = 'notspecified'

    redinds = range(1, len(tmesh), compress)

    redina = np.array(redinds)

    fig = plt.figure(fignum)
    ax1 = fig.add_subplot(111)
    ax1.plot(np.array(tmesh)[redina], np.array(outsig)[redina],
             color='r', linewidth=2.0)

    if tikzfile is not None:
        try:
            from matplotlib2tikz import save as tikz_save
            tikz_save(tikzfile + '.tikz',
                      figureheight='\\figureheight',
                      figurewidth='\\figurewidth'
                      )
            print('tikz saved to ' + tikzfile + '.tikz')
        except ImportError:
            print('`matplotlib2tikz` not found')
    if tikzonly:
        return
    else:
        fig.show()
        return


def plot_outp_sig(str_to_json=None, tmeshkey='tmesh', sigkey='outsig',
                  outsig=None, tmesh=None, fignum=222, reference=None,
                  compress=5, tikzfile=None):
    import matplotlib.pyplot as plt

    if str_to_json is not None:
        jsdict = load_json_dicts(str_to_json)
        tmesh = jsdict[tmeshkey]
        outsig = jsdict[sigkey]
    else:
        str_to_json = 'notspecified'

    redinds = range(1, len(tmesh), compress)
    redina = np.array(redinds)

    NY = len(outsig[0])/2

    fig = plt.figure(fignum)
    ax1 = fig.add_subplot(111)
    ax1.plot(np.array(tmesh)[redina], np.array(outsig)[redina, :NY],
             color='b', linewidth=2.0)
    ax1.plot(np.array(tmesh)[redina], np.array(outsig)[redina, NY:],
             color='r', linewidth=2.0)

    if tikzfile is not None:
        try:
            from matplotlib2tikz import save as tikz_save
            tikz_save(tikzfile + '.tikz',
                      figureheight='\\figureheight',
                      figurewidth='\\figurewidth'
                      )
            print('tikz saved to ' + tikzfile + '.tikz')
        except ImportError:
            print('`matplotlib2tikz` not found')
        fig.show()

    if reference is not None:
        fig = plt.figure(fignum+1)
        ax1 = fig.add_subplot(111)
        ax1.plot(tmesh, np.array(outsig)-reference)

        if tikzfile is not None:
            tikz_save(tikzfile + '_difftoref.tikz',
                      figureheight='\\figureheight',
                      figurewidth='\\figurewidth'
                      )
        fig.show()


def writevp_paraview(velvec=None, pvec=None, strtojson=None, visudict=None,
                     vfile='vel__.vtu', pfile='p__.vtu', ignore_bcs=False):
    if visudict is None:
        #jsfile = file(strtojson)
        jsfile = open(strtojson)
        visudict = json.load(jsfile)
        vaux = np.zeros((visudict['vdim'], 1))
        # fill in the boundary values
        if ignore_bcs:
            pass
        else:
            for bcdict in visudict['bclist']:
                intbcidx = [int(bci) for bci in bcdict.keys()]
                vaux[intbcidx, 0] = list(bcdict.values())
        vaux[visudict['invinds']] = velvec

    vxvtxdofs = visudict['vxvtxdofs']
    vyvtxdofs = visudict['vyvtxdofs']

    #velfile = file(vfile, 'w')
    velfile = open(vfile, 'w')
    velfile.write(visudict['vtuheader_v'])
    #for xvtx, yvtx in itertools.izip(vxvtxdofs, vyvtxdofs):
    for xvtx, yvtx in zip(vxvtxdofs, vyvtxdofs):
        #print(u'{0} {1} {2} '.format(vaux[xvtx][0], vaux[yvtx][0], 0.))
        velfile.write(u'{0} {1} {2} '.format(vaux[xvtx][0], vaux[yvtx][0], 0.))
    velfile.write(visudict['vtufooter_v'])

    if pvec is not None:
        pvtxdofs = visudict['pvtxdofs']
        #pfile = file(pfile, 'w')
        pfile = open(pfile, 'w')
        pfile.write(visudict['vtuheader_p'])
        for pval in pvec[pvtxdofs, 0]:
            pfile.write(u'{0} '.format(pval))
        pfile.write(visudict['vtufooter_p'])


def collect_vtu_files(filelist, pvdfilestr):

    #colfile = file(pvdfilestr, 'w')
    colfile = open(pvdfilestr, 'w')
    colfile.write(u'<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1"> <Collection>\n')
    for tsp, vtufile in enumerate(filelist):
        dtst = u'<DataSet timestep="{0}" part="0" file="{1}"/>'.format(tsp, vtufile)
        colfile.write(dtst)

    colfile.write(u'</Collection> </VTKFile>')
