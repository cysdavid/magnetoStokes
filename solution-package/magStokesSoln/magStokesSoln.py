import numpy as np
from scipy.special import i1,k1

class magStokes():
    '''
    Class for evaluating analytical solutions for magneto-Stokes
    flow in a shallow annulus, given experimental parameters.

    C.S. David, E.W. Hester, Y. Xu, J.M. Aurnou.
    Magneto-Stokes Flow in a Shallow Free-Surface Annulus. In prep.
    '''

    def __init__(self,h,r_i,r_o,B,Imax,h_sigma=0,r_i_sigma=0,r_o_sigma=0,B_sigma=0,Imax_sigma=0):
        '''
        Initializes a magStokes instance.
        Computes aspect ratios χ = r_i/r_o and γ = h/r_o

        Parameters
        ----------
        h : float
            Depth of fluid layer (m)
        r_i : float
            Radius of inner cylinder (m)
        r_o : float
            Radius of outer cylinder (m)
        B : float
            Average magnetic field strength (T)
        Imax : float
            Maximum current through system (A)
        h_sigma : float
            Measurement uncertainty in depth of fluid layer (m)
        r_i_sigma : float
            Measurement uncertainty in radius of inner cylinder (m)
        r_o_sigma : float
            Measurement uncertainty in radius of outer cylinder (m)
        B_sigma : float
            Measurement uncertainty in average magnetic field strength (T)
        Imax_sigma : float
            Measurement uncertainty in maximum current through system (A)

        Reference
        ---------
        C.S. David, E.W. Hester, Y. Xu, J.M. Aurnou.
        Magneto-Stokes Flow in a Shallow Free-Surface Annulus. In prep.
        
        '''
        self.h,self.r_i,self.r_o,self.B,self.Imax = [h,r_i,r_o,B,Imax]
        self.h_sigma,self.r_i_sigma,self.r_o_sigma,self.B_sigma,self.Imax_sigma = [h_sigma,r_i_sigma,r_o_sigma,B_sigma,Imax_sigma]

        # Compute curvature aspect ratio, χ
        self.χ = self.r_i/self.r_o

        # Compute depth aspect ratio, γ
        self.γ = self.h/self.r_o

    def customFluidProps(self,mu,rho,mu_sigma=0,rho_sigma=0):
        '''
        Adds dynamic viscosity (Pa s) and density (kg/m^3)
        of an arbitrary fluid  to the magStokes instance.
        Computes kinematic viscosity (m^2/s), magneto-Stokes
        velocity scale U (m/s), advective time scale T (s), circulation
        timescale Tcirc (s), spin-up timescale Tsp (s), and control
        Reynolds number ReC.

        Parameters
        ----------
        S : float
            Salinity in g/L = ppt mass:volume
        S_sigma : float
            Uncertainty in salinity in g/L
        T_sigma: float
            Uncertainty in temperature in Celsius
    
        '''
        self.mu = mu
        self.mu_sigma = mu_sigma
        self.rho = rho
        self.rho_sigma = rho_sigma
        self.nu = self.mu/self.rho
        self.U = (self.Imax*self.B/(2*np.pi*self.mu)) * (self.γ/(1+self.χ))
        self.T = (self.r_i + self.r_o)/self.U
        self.Tcirc = np.pi*self.T
        self.Tsp = (2/np.pi)**2 * self.h**2/self.nu
        self.ReC = (self.h**2/self.nu)/((self.r_o+self.r_i)/self.U)

    def saltwaterProps(self,S,S_sigma=0,T_sigma=0):
        '''
        Computes dynamic viscosity (Pa s) and density (kg/m^3)
        of seawater at 20°C from salinity, using the fittings
        provided by Isdale & Morris (1972) and Isdale et al. (1972).
        Adds these values to the magStokes instance. Computes kinematic
        viscosity (m^2/s), magneto-Stokes velocity scale U (m/s),
        advective time scale T (s), circulation timescale Tcirc (s),
        spin-up timescale Tsp (s), and control Reynolds number ReC.

        Parameters
        ----------
        S : float
            Salinity in g/L = ppt mass:volume
        S_sigma : float
            Uncertainty in salinity in g/L
        T_sigma: float
            Uncertainty in temperature in Celsius

        Reference
        ---------
        J.D. Isdale, R. Morris,
        Physical properties of sea water solutions: density,
        Desalination, Volume 10, Issue 4, 1972, Pages 329-339, ISSN 0011-9164,
        https://doi.org/10.1016/S0011-9164(00)80003-X.

        J.D. Isdale, C.M. Spence, J.S. Tudhope,
        Physical properties of sea water solutions: viscosity,
        Desalination, Volume 10, Issue 4, 1972, Pages 319-328, ISSN 0011-9164,
        https://doi.org/10.1016/S0011-9164(00)80002-8.
        '''
        self.mu, self.mu_sigma = magStokes.mu_func(S,S_sigma,T_sigma)
        self.rho, self.rho_sigma = magStokes.rho_func(S,S_sigma,T_sigma)
        self.nu = self.mu/self.rho
        self.U = (self.Imax*self.B/(2*np.pi*self.mu)) * (self.γ/(1+self.χ))
        self.T = (self.r_i + self.r_o)/self.U
        self.Tcirc = np.pi*self.T
        self.Tsp = (2/np.pi)**2 * self.h**2/self.nu
        self.ReC = (self.h**2/self.nu)/((self.r_o+self.r_i)/self.U)

    def mu_func(S,S_sigma,T_sigma):
        '''
        Computes dynamic viscosity of seawater at 20°C
        from salinity using results of Isdale et al. (1972).

        Parameters
        ----------
        S : float
            Salinity in g/L = ppt mass:volume
        S_sigma : float
            Uncertainty in salinity in g/L
        T_sigma: float
            Uncertainty in temperature in Celsius
        
        Returns
        -------
        mu : float
            Dynamic viscosity in Pa s
        mu_sigma : float
            Uncertainty in dynamic viscosity in Pa s

        Reference
        ---------
        J.D. Isdale, C.M. Spence, J.S. Tudhope,
        Physical properties of sea water solutions: viscosity,
        Desalination, Volume 10, Issue 4, 1972, Pages 319-328, ISSN 0011-9164,
        https://doi.org/10.1016/S0011-9164(00)80002-8.
        '''
        T = 20 # °C

        # Convert from mass:volume to mass:mass salinity
        # g/kg salinity = (1L/0.9982 kg) g/L salinity at 20°C
        Sm = S/0.9982 
        Sm_sigma = S_sigma/0.9982
        
        # fitting constants:
        A = 1.37220 
        a1 = -0.001015 
        a2 = 0.000005 
        B = 0.000813 
        b1 = 0.006102
        b2 = -0.000040
        c1 = 0.001550
        c2 = 0.0000093

        mu = 1e-3*(1.002 + c1*Sm + c2*Sm**2)/10**(
            ((A*(1 + a1*Sm + a2*Sm**2) + B*(1 + b1*Sm + b2*Sm**2)*(-20 + T))*(-20 + T))/(109 + T)) # Pa s
        
        mu_sigma = 1e-3*np.sqrt(((1.002 + c1*Sm + c2*Sm**2)**2*(
            A*(129. + 129.*a1*Sm + 129.*a2*Sm**2) + B*(
            -4760. + 218.*T + T**2 + b1*Sm*(-4760. + 218.*T + T**2) + b2*Sm**2*(
            -4760. + 218.*T + T**2)))**2*T_sigma**2*np.log(10)**2)/(109. + T)**4 + Sm_sigma**2*(
            c1 + 2*c2*Sm - ((1.002 + c1*Sm + c2*Sm**2)*(A*(a1 + 2*a2*Sm) + B*(b1 + 2*b2*Sm)*(-20 + T)
            )*(-20 + T)*np.log(10))/(109 + T))**2)/10**(((A*(1 + a1*Sm + a2*Sm**2) + B*(
            1 + b1*Sm + b2*Sm**2)*(-20 + T))*(-20 + T))/(109 + T))
        
        return mu, mu_sigma
    
    def rho_func(S,S_sigma,T_sigma):
        '''
        Computes density of seawater at 20°C
        from salinity using results of Isdale & Morris (1972).

        Parameters
        ----------
        S : float
            Salinity in g/L = ppt mass:volume
        S_sigma : float
            Uncertainty in salinity in g/L
        T_sigma: float
            Uncertainty in temperature in Celsius
        
        Returns
        -------
        rho : float
            Density in kg/m^3
        rho_sigma : float
            Uncertainty in density in kg/m^3

        Reference
        ---------
        J.D. Isdale, R. Morris,
        Physical properties of sea water solutions: density,
        Desalination, Volume 10, Issue 4, 1972, Pages 329-339, ISSN 0011-9164,
        https://doi.org/10.1016/S0011-9164(00)80003-X.
        '''
        # Convert from mass:volume to mass:mass salinity
        # g/kg salinity = (1L/0.9982 kg) g/L salinity at 20°C
        Sm = S/0.9982 
        Sm_sigma = S_sigma/0.9982
        T = 20 # °C
        A = (2*T-200)/160
        F1 = 0.5
        F2 = A
        F3 = 2*A**2 - 1
        F4 = 4*A**3 - 3*A
        B = (2*Sm-150)/150
        G1 = 0.5
        G2 = B
        G3 = 2*B**2 - 1
        A1 = 4.032*G1 + 0.115*G2 + 3.26e-4*G3
        A2 = -0.108*G1 + 1.571e-3*G2 -4.23e-4*G3
        A3 = -0.012*G1 + 1.74e-3*G2 - 9e-6*G3
        A4 = 6.92e-4*G1 - 8.7e-5*G2 - 5.3e-5*G3

        rho = 1e3*(A1*F1 + A2*F2 + A3*F3 + A4*F4)
        rho_sigma = 1000*np.sqrt(Sm_sigma**2*(0.0007471391666666668 - 6.080416666666665e-7*T + 3.4937499999999994e-9*T**2 + 1.3020833333333335e-11*T**3 + Sm*(6.314222222222222e-7 - 1.0780000000000001e-8*T + 8.633333333333333e-11*T**2 - 2.9444444444444447e-13*T**3))**2 + (-0.00014080000000000006 - 6.624375e-6*T + 8.90625e-9*T**2 + Sm**2*(-5.390000000000001e-9 + 8.633333333333333e-11*T - 4.4166666666666674e-13*T**2) + Sm*(-6.080416666666666e-7 + 6.987499999999999e-9*T + 3.906250000000001e-11*T**2))**2*T_sigma**2)
        
        return rho, rho_sigma
    
    def __str__(self):
        if hasattr(self,"mu"):
            return f"χ = {self.χ:.2f}, γ = {self.γ:.2f}, ReC = {self.ReC:.2f}, U = {self.U:.2e} m/s"
        else:
            return f"χ = {self.χ:.2f}, γ = {self.γ:.2f}"
    
    def uShallowND(self,rND,zND):
        '''
        Computes the nondimensional steady
        velocity in the shallow limit (γ -> 0).

        Parameters
        ----------
        rND : float
            Dimensionless radial position
            rND = r/r_o.
        zND : float
            Dimensionless vertical distance
            from the bottom of the channel
            zND = z/h.

        Returns
        -------
        uNdim : float
            Dimensionless velocity uNdim = u/U,
            where U = (Imax B/(2 π μ)) * (γ/(1+χ)).

        Reference
        ---------
        C.S. David, E.W. Hester, Y. Xu, J.M. Aurnou.
        Magneto-Stokes Flow in a Shallow Free-Surface Annulus. In prep.
        '''
        uND = (self.χ + 1)/2*(2*zND - zND**2)/rND
        return uND

    def uSteadyND(self,rND,zND,l=3):
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
        l : int
            The number of terms to retain in 
            the 2D series solution.

        Returns
        -------
        uNdim : float
            Dimensionless velocity uNdim = u/U,
            where U = (Imax B/(2 π μ)) * (γ/(1+χ)).

        Reference
        ---------
        C.S. David, E.W. Hester, Y. Xu, J.M. Aurnou.
        Magneto-Stokes Flow in a Shallow Free-Surface Annulus. In prep.
        '''
        u2Darr = np.array([magStokes.An(rND,n,self.γ,self.χ)*np.sin((-0.5 + n)*np.pi*zND) for n in range(1,l+1)])
        return u2Darr.sum(axis=0)

    def uND(self,rND,zND,tND,l=3):
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
        l : int
            The number of terms to retain in
            the 2D series solution.

        Returns
        -------
        uNdim : float
            Dimensionless velocity uNdim = u/U,
            where U = (Imax B/(2 π μ)) * (γ/(1+χ)).

        Reference
        ---------
        C.S. David, E.W. Hester, Y. Xu, J.M. Aurnou.
        Magneto-Stokes Flow in a Shallow Free-Surface Annulus. In prep.
        '''
        uNdim = self.uSteadyND(rND,zND,l)*(1-np.exp(-np.pi**2*tND/(4*self.ReC)))
        return uNdim
    
    def thetaShallowND(self,rND,zND,tND,θ0=0):
        '''
        Computes the angular position of a fluid parcel
        starting at θ = θ0, dimensionless radial position
        rND, and dimensionless vertical position zND, for an
        initially-quiescent layer of vanishing depth (h/r_o --> 0)
        at dimensionless time tND after current is turned on.

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
        θ0 : float
            Initial angular position
            of fluid parcel (rad).

        Returns
        -------
        θ : float
            Angular position (rad) of fluid
            parcel at dimensionless time tND.

        Reference
        ---------
        C.S. David, E.W. Hester, Y. Xu, J.M. Aurnou.
        Magneto-Stokes Flow in a Shallow Free-Surface Annulus. In prep.
        '''
        uShal = self.uShallowND(rND,zND)*(1-np.exp(-np.pi**2*tND/(4*self.ReC)))
        θ = θ0 + (uShal*(4*(-1 + np.exp(-(np.pi**2*tND)/(4*self.ReC)))*self.ReC + np.pi**2*tND)*(1 + self.χ))/(np.pi**2*rND)
        return θ
    
    def thetaND(self,rND,zND,tND,θ0=0,l=3):
        '''
        Computes the angular position of a fluid parcel
        starting at θ = θ0, dimensionless radial position
        rND, and dimensionless vertical position zND, for an
        initially-quiescent layer of finite depth (h/r_o > 0)
        at dimensionless time tND after current is turned on.

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
        θ0 : float
            Initial angular position
            of fluid parcel (rad).
        l : int
            The number of terms to retain in
            the 2D series solution.

        Returns
        -------
        θ : float
            Angular position (rad) of fluid
            parcel at dimensionless time tND.

        Reference
        ---------
        C.S. David, E.W. Hester, Y. Xu, J.M. Aurnou.
        Magneto-Stokes Flow in a Shallow Free-Surface Annulus. In prep.
        '''
        u2D = self.uND(rND,zND,tND,l)
        θ = θ0 + (u2D*(4*(-1 + np.exp(-(np.pi**2*tND)/(4*self.ReC)))*self.ReC + np.pi**2*tND)*(1 + self.χ))/(np.pi**2*rND)
        return θ
    
    def uShallow(self,r,z):
        '''
        Computes the (dimensional) steady
        velocity in the shallow limit (γ -> 0).

        Parameters
        ----------
        r : float
            Radial position (m).
        z : float
            Vertical distance from the
            bottom of the channel (m).

        Returns
        -------
        u : float
            Velocity (m/s)

        Reference
        ---------
        C.S. David, E.W. Hester, Y. Xu, J.M. Aurnou.
        Magneto-Stokes Flow in a Shallow Free-Surface Annulus. In prep.
        '''
        rND = r/self.r_o
        zND = z/self.h
        u = self.U*self.uShallowND(rND,zND)
        return u
    
    def uSteady(self,r,z,l=3):
        '''
        Computes the (dimensional) steady velocity
        for a layer of finite depth (h/r_o > 0).

        Parameters
        ----------
        r : float
            Radial position (m).
        z : float
            Vertical distance from the
            bottom of the channel (m).
        l : int
            The number of terms to retain in 
            the 2D series solution.

        Returns
        -------
        u : float
            Velocity (m/s)

        Reference
        ---------
        C.S. David, E.W. Hester, Y. Xu, J.M. Aurnou.
        Magneto-Stokes Flow in a Shallow Free-Surface Annulus. In prep.
        '''
        rND = r/self.r_o
        zND = z/self.h
        u = self.U*self.uSteadyND(rND,zND,l)
        return u

    def u(self,r,z,t,l=3):
        '''
        Computes the (dimensional) time-varying
        velocity for an initially-quiescent layer
        of finite depth (h/r_o > 0) at time t
        after current is turned on.

        Parameters
        ----------
        r : float
            Radial position (m).
        z : float
            Vertical distance from the
            bottom of the channel (m).
        t : float
            Time after electrical power
            supply is turned on (s).
        l : int
            The number of terms to retain in
            the 2D series solution.

        Returns
        -------
        u : float
            Velocity (m/s)

        Reference
        ---------
        C.S. David, E.W. Hester, Y. Xu, J.M. Aurnou.
        Magneto-Stokes Flow in a Shallow Free-Surface Annulus. In prep.
        '''
        rND = r/self.r_o
        zND = z/self.h
        tND = t/self.T
        u = self.U*self.uSteadyND(rND,zND,l)*(1-np.exp(-np.pi**2*tND/(4*self.ReC)))
        return u
    
    def thetaShallow(self,r,z,t,θ0=0):
        '''
        Computes the angular position of a fluid parcel
        starting at θ = θ0, radial position r, and vertical
        position z, for an initially-quiescent layer of vanishing
        depth (h/r_o --> 0) at time t after current is turned on.

        Parameters
        ----------
        r : float
            Radial position (m).
        z : float
            Vertical distance from the
            bottom of the channel (m).
        t : float
            Time after electrical power
            supply is turned on (s).
        θ0 : float
            Initial angular position
            of fluid parcel (rad).

        Returns
        -------
        θ : float
            Angular position (rad)
            of fluid parcel at time t.

        Reference
        ---------
        C.S. David, E.W. Hester, Y. Xu, J.M. Aurnou.
        Magneto-Stokes Flow in a Shallow Free-Surface Annulus. In prep.
        '''
        rND = r/self.r_o
        zND = z/self.h
        tND = t/self.T
        θ = self.thetaShallowND(rND,zND,tND,θ0)
        return θ
    
    def theta(self,r,z,t,θ0=0,l=3):
        '''
        Computes the angular position of a fluid parcel
        starting at θ = θ0, radial position r, and vertical
        position z, for an initially-quiescent layer of finite
        depth (h/r_o > 0) at time t after current is turned on.

        Parameters
        ----------
        r : float
            Radial position (m).
        z : float
            Vertical distance from the
            bottom of the channel (m).
        t : float
            Time after electrical power
            supply is turned on (s).
        θ0 : float
            Initial angular position
            of fluid parcel (rad).
        l : int
            The number of terms to retain in
            the 2D series solution.

        Returns
        -------
        θ : float
            Angular position (rad)
            of fluid parcel at time t.

        Reference
        ---------
        C.S. David, E.W. Hester, Y. Xu, J.M. Aurnou.
        Magneto-Stokes Flow in a Shallow Free-Surface Annulus. In prep.
        '''
        rND = r/self.r_o
        zND = z/self.h
        tND = t/self.T
        θ = self.thetaND(rND,zND,tND,θ0,l)
        return θ
    
    def An(r,n,γ,χ):
        an = (16*(1 + χ)*(i1(((-0.5 + n)*np.pi*r)/γ)*(-(r*k1(((-0.5 + n)*np.pi)/γ)) + χ*r*k1((χ*(-0.5 + n)*np.pi)/γ)) + 
                χ*i1((χ*(-0.5 + n)*np.pi)/γ)*(k1(((-0.5 + n)*np.pi)/γ) - r*k1(((-0.5 + n)*np.pi*r)/γ)) + 
                i1(((-0.5 + n)*np.pi)/γ)*(-(χ*k1((χ*(-0.5 + n)*np.pi)/γ)) + 
                r*k1(((-0.5 + n)*np.pi*r)/γ))))/(χ*(-1 + 2*n)**3*np.pi**3*r*(i1((χ*(-0.5 + n)*np.pi)/γ)*k1(((-0.5 + n)*np.pi)/γ) - 
                i1(((-0.5 + n)*np.pi)/γ)*k1((χ*(-0.5 + n)*np.pi)/γ)))
        return an



