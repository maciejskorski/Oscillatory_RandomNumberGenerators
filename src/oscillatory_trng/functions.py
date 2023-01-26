import numpy as np
import mpmath as mpm

def jtheta3(z,q):
    """
        Jacobi Theta-3 function.
        The scaling convention follows the NIST definition (dlmf.nist.gov) and Abramowitz, M. & Stegun, I. A. (1964), "Handbook of Mathematical Functions... ".
    """
    def fn(z,q):
        jtheta3_fn = lambda z,q: mpm.jtheta(n=3,z=z,q=q)
        jtheta3_fn = np.vectorize(jtheta3_fn,otypes=(np.cfloat,))
        return jtheta3_fn(z,q)
    
    return fn(z,q)


def brownian_mod1_logp(mu,sigma,t,y):
    """Evaluate the log-probability of the Brownian motion modulo 1.
    Args:
        mu: drift parameter of the Brownian motion
        sigma: drift parameter of the Brownian motion
        t: 1-D array of time points; non-decreasing and positive
        y: array of points to evaluate density; values between 0 and 1, first dimension matches the length of t.
    """
    pads_dim = [(1,0)]+(len(y.shape)-1)*[(0,0)]
    t = np.expand_dims(t,list(np.arange(1,len(y.shape))))
    t = np.pad(t,pads_dim,mode='constant')
    y = np.pad(y,pads_dim,mode='constant')
    t_diff = np.diff(t,axis=0)
    y_diff = np.diff(y,axis=0)
    z = np.pi*(y_diff-mu*t_diff)
    q = np.exp(-2*np.pi**2*t_diff*sigma**2)
    return np.log(jtheta3(z,q)).sum(0)
    
