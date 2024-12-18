import numpy as np
from scipy.special import i1,k1

def An(r,n,γ,χ):
    an = (16*(1 + χ)*(i1(((-0.5 + n)*np.pi*r)/γ)*(-(r*k1(((-0.5 + n)*np.pi)/γ)) + χ*r*k1((χ*(-0.5 + n)*np.pi)/γ)) + 
            χ*i1((χ*(-0.5 + n)*np.pi)/γ)*(k1(((-0.5 + n)*np.pi)/γ) - r*k1(((-0.5 + n)*np.pi*r)/γ)) + 
            i1(((-0.5 + n)*np.pi)/γ)*(-(χ*k1((χ*(-0.5 + n)*np.pi)/γ)) + 
            r*k1(((-0.5 + n)*np.pi*r)/γ))))/(χ*(-1 + 2*n)**3*np.pi**3*r*(i1((χ*(-0.5 + n)*np.pi)/γ)*k1(((-0.5 + n)*np.pi)/γ) - 
            i1(((-0.5 + n)*np.pi)/γ)*k1((χ*(-0.5 + n)*np.pi)/γ)))
    return an

def uSteadyND(rND,zND,γ,χ,l=3):
    '''
    Computes the nondimensional steady velocity
    for a layer of finite depth (γ > 0).

    Parameters
    ----------
    rND : float
        Dimensionless radial position
        rND = r/r_o.
    zND : float
        Dimensionless vertical distance
        from the bottom of the channel
        zND = z/h.
    γ : float
        Aspect ratio
        γ = h/r_o
    χ : float
        Radius ratio
        χ = r_i/r_o
    l : int, optional
        The number of terms to retain in 
        the 2D series solution. By default,
        3 terms will be used.

    Returns
    -------
    uNdim : float
        Dimensionless steady-state velocity uNdim = u/U,
        where U = (Imax B/(2 π μ)) * (γ/(1+χ)).

    Reference
    ---------
    C.S. David, E.W. Hester, Y. Xu, J.M. Aurnou.
    Magneto-Stokes Flow in a Shallow Free-Surface Annulus. J. Fluid Mech. (2024). 
    https://doi.org/10.1017/jfm.2024.674
    '''
    u2Darr = np.array([An(rND,n,γ,χ)*np.sin((-0.5 + n)*np.pi*zND) for n in range(1,l+1)])
    return u2Darr.sum(axis=0)

def uND(rND,zND,tND,γ,χ,ReC,l=3):
    '''
    Computes the nondimensional time-varying
    velocity for an initially-quiescent layer
    of finite depth (γ > 0) at dimensionless
    time tND after current is turned on.

    Parameters
    ----------
    rND : float
        Dimensionless radial position
        rND = r/r_o.
    zND : float
        Dimensionless vertical distance
        from the bottom of the channel
        zND = z/h.
    tND : float
        Dimensionless time after electrical power
        supply is turned on: tND = t/((r_o + r_i)/U),
        where U = (Imax B/(2 π μ)) * (γ/(1+χ)).
    γ : float
        Aspect ratio
        γ = h/r_o
    χ : float
        Radius ratio
        χ = r_i/r_o
    ReC : float
        Control Reynolds number
    l : int, optional
        The number of terms to retain in
        the 2D series solution. By default,
        3 terms will be used.

    Returns
    -------
    uNdim : float
        Dimensionless velocity uNdim = u/U,
        where U = (Imax B/(2 π μ)) * (γ/(1+χ)).

    Reference
    ---------
    C.S. David, E.W. Hester, Y. Xu, J.M. Aurnou.
    Magneto-Stokes Flow in a Shallow Free-Surface Annulus. J. Fluid Mech. (2024).
    https://doi.org/10.1017/jfm.2024.674
    '''
    uNdim = uSteadyND(rND,zND,γ,χ,l)*(1-np.exp(-np.pi**2*tND/(4*ReC)))
    return uNdim