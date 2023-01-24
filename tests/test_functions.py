from scipy.stats import norm
import numpy as np
from oscillatory_trng.functions import jtheta3

def test_1d_probability():
    """ Test the formula expressing normal distribution modulo 1 in terms of Jacobi theta function. """
    from oscillatory_trng.functions import jtheta3

    # try example values
    sigma = 0.25
    mu = 0.1
    y = 0.05
    # by definition
    pdf = norm(mu,sigma).pdf
    ks = np.arange(-100,100)
    val_def = pdf(y+ks).sum()
    # by theorem
    z = np.pi*(y-mu)
    q = np.exp(-2*sigma**2*np.pi**2)
    val_thm = jtheta3(z,q)

    np.testing.assert_almost_equal(val_def, val_thm, decimal=5)
    