""""Ploting tools"""

__all__ = [
            "plotting_SVD_decay",
            "plotting_obv_vel",
            "plotting_abs_error",
            "print_nparray_tex"
          ]

import matplotlib.pyplot as plt
import numpy as np

## Ploting SVD decay
################################

def plotting_SVD_decay(S, string = 'velocity'):
    fig = plt.figure()
    ax = fig.add_subplot(111)    # The big subplot
    ax.semilogy(S,label = string)
    # Set common labels
    ax.set_xlabel('k')
    ax.set_ylabel('singular values')
    plt.legend()
    

def plotting_obv_vel(V, Voptinf, Vpod, Vdmdc, Vdmd_quad, T, Cv):
    plt.figure()
    plt.plot(T, (Cv@V).T, 'b', label='original')
    plt.plot(T, (Cv@Voptinf).T, '--g', label='optinf')
    plt.plot(T, (Cv@Vpod).T, '--r', label='POD')
    plt.plot(T, (Cv@Vdmdc).T,'--y', label = 'DMDc')
    plt.plot(T, (Cv@Vdmd_quad).T,'--m', label = 'DMD_quad')
    #plt.legend(loc='best')
    plt.xlabel('t')
    

def plotting_abs_error(V, Vapprox, T, string = 'title'):
    plt.figure()
    ax = plt.subplot(111)
    ax.semilogy(T,np.mean(np.abs(Vapprox - V).T,axis=1))
    plt.title(string)
    plt.xlabel('t')


def print_nparray_tex(array, formatit='math', fstr='.4f', name=None):
    tdarray = np.atleast_2d(array)
    if name is not None:
        print(*name, sep=' & ')
    if formatit is None:
        print(" \\\\\n".join([" & ".join(map(('{0:' + fstr + '}').
                             format, line))
                             for line in tdarray]))
    elif formatit == 'math':
        print(" \\\\\n".join([" & ".join(map(('${0:' + fstr + '}$').
                                         format, line))
                             for line in tdarray]))
    else:
        fsb = '\\' + formatit + '{{'
        fse = '}}'

        print(" \\\\\n".join([" & ".join(map((fsb + '{0:' + fstr + '}' + fse).
                             format, line))
                             for line in tdarray]))
