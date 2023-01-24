import numpy as np
import mpmath as mpm

def jtheta3(z,q):
    """
        Jacobi Theta-3 function, as defined in Abramowitz, M. & Stegun, I. A. (1964), Handbook of Mathematical Functions with Formulas, Graphs, and Mathematical Tables
    """
    def jtheta3_fn(z,q):
        jtheta3_fn = lambda z,q: mpm.jtheta(n=3,z=z,q=q)
        jtheta3_fn = np.vectorize(jtheta3_fn,otypes=(float,))
        return jtheta3_fn(z,q)
    
    return jtheta3_fn(z,q)

