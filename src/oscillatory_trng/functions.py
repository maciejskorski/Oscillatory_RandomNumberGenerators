import numpy as np
import mpmath as mpm

def jtheta3(z,q):
    """
        Jacobi Theta-3 function.
        The scaling convention follows the definition in Abramowitz, M. & Stegun, I. A. (1964), "Handbook of Mathematical Functions... ".
    """
    def jtheta3_fn(z,q):
        jtheta3_fn = lambda z,q: mpm.jtheta(n=3,z=z,q=q)
        jtheta3_fn = np.vectorize(jtheta3_fn,otypes=(float,))
        return jtheta3_fn(z,q)
    
    return jtheta3_fn(z,q)


def browian_mod1_logp(mu,sigma,t,y):
    """Evaluate the log-probability of the Brownian motion modulo 1.
    Args:
        mu: drift parameter of the Brownian motion
        sigma: drift parameter of the Brownian motion
        t: array of time points, non-decreasign and non-negative
        y: array of points to evaluate density, between 0 and 1
    """
    t_diff = np.diff(np.append(0,t))
    y_diff = np.diff(np.append(0,y))
    z = np.pi*(y_diff-mu*t_diff)
    q = np.exp(-2*np.pi**2*t_diff*sigma**2)
    return np.log(jtheta3(z,q)).sum()
    
