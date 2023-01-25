import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.special import logsumexp


def test_jtheta3():
    """Test the implementation of 3rd theta function"""
    from oscillatory_trng.functions import jtheta3
    import mpmath as mp
    # try a complex z-zero of theta function, cf dlmf.nist.gov
    sigma = 0.25
    q =np.exp(-2*np.pi**2*sigma**2)
    z0 = np.pi*(1/2+1/2*2*np.pi*sigma**2*1j)
    val_formula = jtheta3(z0,q)
    val_true = 0

    np.testing.assert_almost_equal(val_true, val_formula)



def test_1d_probability():
    """Test the formula expressing normal distribution modulo 1 in terms of Jacobi theta function."""
    from oscillatory_trng.functions import jtheta3
    # try example values
    sigma = 0.25
    mu = 0.1
    y = 0.05
    # by definition
    dist = norm(mu, sigma)
    ks = np.arange(-100, 100)
    logp_true = logsumexp(dist.logpdf(y + ks))
    # by theorem
    z = np.pi * (y - mu)
    q = np.exp(-2 * sigma**2 * np.pi**2)
    logp_formula = np.log(jtheta3(z, q))

    np.testing.assert_almost_equal(logp_true, logp_formula)


def test_browian_mod1_logp():
    """Test the formula for finite-dimentional probability of Brownian motion modulo 1"""
    from oscillatory_trng.functions import brownian_mod1_logp

    # try example values
    sigma = 0.25
    mu = 0.01
    t = np.array([1, 2.5, 3])
    y = np.array([0.2, 0.5, 0.3])
    mean = mu * t
    cov = sigma**2 * np.clip(t, 0, t.reshape(-1, 1))
    dist = multivariate_normal(mean, cov)
    ks = np.arange(-10, 10)
    ks = np.meshgrid(ks, ks, ks)
    ks = np.stack(ks, axis=-1)  # (N,N,N,3)
    logp_true = logsumexp(dist.logpdf(y + ks))

    logp_formula = brownian_mod1_logp(mu, sigma, t, y)

    np.testing.assert_almost_equal(logp_true, logp_formula)
