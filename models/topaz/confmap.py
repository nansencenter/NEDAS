###adapted from NERSC-HYCOM-CICE/pythonlibs/modelgrid
import numpy as np

# some constants
_pi_1=np.pi
_pi_2=_pi_1/2.
_deg=180./_pi_1
_rad=1.0/_deg
_epsil=1.0E-9

class ConformalMapping(object) :
    def __init__(self,lat_a,lon_a,lat_b,lon_b,
            wlim,elim,ires,
            slim,nlim,jres,
            mercator,
            mercfac,lold) :
        """Constructor: arguments: (grid.info)
            lat_a, lon_a      : position of pole A in geo coordinates
            lat_b, lon_b      : position of pole B in geo coordinates
            wlim, elim, ires : western,  eastern  limits in new coords and number of points in 1st dim
            slim, nlim, jres : southern, northern limits in new coords and number of points in 2nd dim
            mercator
            mercfac, lold
        """
        self._lat_a = lat_a
        self._lon_a = lon_a
        self._lat_b = lat_b
        self._lon_b = lon_b

        self._wlim = wlim
        self._elim = elim
        self._ires = ires

        self._slim = slim
        self._nlim = nlim
        self._jres = jres

        self._mercator = mercator
        self._mercfac  = mercfac
        self._lold     = lold

        self._di=(self._elim-self._wlim)/float(ires-1)    # delta lon'
        self._dj=(self._nlim-self._slim)/float(jres-1)    # delta lat' for spherical grid

        if self._mercator:
            self._dj=self._di
            self._dm=self._di
            if lold:
                self._slim=-self._mercfac*self._jres*self._dj
            else:
                self._slim= self._mercfac

        # transform to spherical coordinates
        self._theta_a=self._lon_a*_rad
        self._phi_a=_pi_2-self._lat_a*_rad
        self._theta_b=self._lon_b*_rad
        self._phi_b=_pi_2-self._lat_b*_rad

        # find the angles of a vector pointing at a point located exactly
        # between the poles
        cx=np.cos(self._theta_a)*np.sin(self._phi_a)+np.cos(self._theta_b)*np.sin(self._phi_b)
        cy=np.sin(self._theta_a)*np.sin(self._phi_a)+np.sin(self._theta_b)*np.sin(self._phi_b)
        cz=np.cos(self._phi_a)+np.cos(self._phi_b)

        theta_c=np.arctan2(cy,cx)
        self._phi_c=_pi_2-np.arctan2(cz,np.sqrt(cx*cx+cy*cy))

        # initialize constants used in the conformal mapping
        self._imag=complex(.0,1.)
        self._ac=np.tan(.5*self._phi_a)*np.exp(self._imag*self._theta_a)
        self._bc=np.tan(.5*self._phi_b)*np.exp(self._imag*self._theta_b)
        self._c =np.tan(.5*self._phi_c)*np.exp(self._imag*theta_c)
        self._cmna=self._c-self._ac
        self._cmnb=self._c-self._bc

        w=self._cmnb/self._cmna
        self._mu_s=np.arctan2(np.imag(w),np.real(w))
        self._psi_s=2.*np.arctan(abs(w))

    @classmethod
    def init_from_file(cls,filename) :
        fid=open(filename,"r")

        #-40.0 140.0      ! Position of pole N (lattitude, longitude):
        tmp=fid.readline().split("!")[0].strip().split()
        lat_a,lon_a = [float(elem) for elem in tmp ]

        #-50.0 140.0        ! Position of pole S (lattitude, longitude):
        tmp=fid.readline().split("!")[0].strip().split()
        lat_b,lon_b = [float(elem) for elem in tmp ]

        #178.2 181.42 800 ! Longitude interval West lon,    East lon, idim
        tmp=fid.readline().split("!")[0].strip().split()
        wlim,elim = [float(elem) for elem in tmp[0:2]]
        ires        = int(tmp[2])

        # 0.0  80.0 760 ! Lattitude interval south limit, north limit, jdim
        tmp=fid.readline().split("!")[0].strip().split()
        slim,nlim = [float(elem) for elem in tmp[0:2]]
        jres        = int(tmp[2])

        #.true.                ! Generate topography
        fid.readline() # Not needed for grid generation

        #.true.                ! dump teclatlon.dat
        fid.readline() # Not needed for grid generation

        #.true.                ! dump micom latlon.dat
        fid.readline() # Not needed for grid generation

        #.true.                ! mercator grid (true, false)
        tmp=fid.readline().split("!")[0].strip().split()
        if tmp[0] == ".true." :
            mercator=True
        elif tmp[0] == ".false." :
            mercator=False
        else :
            raise ValueError("Unable to safely resolve value of mercator flag for confmap")

        #    0.365 .false.    ! merc fac
        tmp=fid.readline().split("!")[0].strip().split()
        mercfac = float(tmp[0])
        if tmp[1] == ".true." :
            lold=True
        elif tmp[1] == ".false." :
            lold=False
        else :
            raise ValueError("Unable to safely resolve value of lold flag for confmap")

        #.false.              ! Smoothing, Shapiro filter
        fid.readline() # Not needed for grid generation

        # 8    2                ! Order of Shapiro filter,  number of passes
        fid.readline() # Not needed for grid generation

        fid.close()

        return cls(lat_a,lon_a,lat_b,lon_b,
                   wlim,elim,ires,
                   slim,nlim,jres,
                   mercator,
                   mercfac,lold)

    def oldtonew(self,lat_o,lon_o) :
        # this routine performes a conformal mapping of the old to the new
        # coordinate system
        lat_o=np.atleast_1d(lat_o)
        lon_o=np.atleast_1d(lon_o)

        # transform to spherical coordinates
        theta=np.mod(lon_o*_rad+3.0*_pi_1,2.0*_pi_1)-_pi_1
        phi=_pi_2-lat_o*_rad

        # transform to the new coordinate system: 1)
        z=np.tan(.5*phi)*np.exp(self._imag*theta)
        w=(z-self._ac)*self._cmnb/((z-self._bc)*self._cmna)
        mu=np.arctan2(np.imag(w),np.real(w))
        psi=2.*np.arctan(abs(w))

        # transform to the new coordinate system: 2)
        I=np.abs(phi-_pi_1)<_epsil
        mu[I]=self._mu_s
        psi[I]=self._psi_s

        # transform to the new coordinate system: 3)
        I=np.logical_and(np.abs(phi-self._phi_b)<_epsil, np.abs(theta-self._theta_b)<_epsil)
        mu[I]=0.
        psi[I]=_pi_1

        # transform to new lat/lon coordinates
        lat_n=(_pi_2-psi)*_deg
        lon_n=mu*_deg

        return lat_n,lon_n

    def newtoold(self,lat_n,lon_n) :
        # this routine performes a conformal mapping of the new to the old 
        # coordinate system

        # transform to spherical coordinates
        mu=np.mod(lon_n*_rad+3*_pi_1,2*_pi_1)-_pi_1
        psi=np.abs(_pi_2-lat_n*_rad)

        # transform to the old coordinate system
        w=np.tan(.5*psi)*np.exp(self._imag*mu)
        z=(self._ac*self._cmnb-w*self._bc*self._cmna)/(self._cmnb-w*self._cmna)
        theta=np.arctan2(np.imag(z),np.real(z))
        phi=2.*np.arctan(np.abs(z))

        I = np.abs(psi-_pi_1) < _epsil
        theta[I]=self._theta_b
        phi  [I]=self._phi_b

        I = np.logical_and(
            np.abs(mu-self._mu_s)<_epsil,
            (psi-self._psi_s)<_epsil)
        theta[I]=0.
        phi  [I]=_pi_1

        # transform to lat/lon coordinates
        lat_o=(_pi_2-phi)*_deg
        lon_o=theta*_deg

        return lat_o,lon_o

    def pivotp(self,lat_n,lon_n) :
        # This subroutine computes the pivot point of each of the observations
        # in the temporary array tmpobs of type observation. The pivot point
        # is the biggest i and the biggest j, (i,j) is the computation points/
        # the grid, that is less than the position of the observation.
        lat_n=np.atleast_1d(lat_n)
        lon_n=np.atleast_1d(lon_n)

        # fix for wrap-around:
        lon_n[np.where(lon_n<0)] += 360.

        ipiv=(lon_n-self._wlim)/self._di + 1.0

        jpiv=np.ones(ipiv.shape)*-999
        if self._mercator:
            I=np.abs(lat_n) < 89.999
            tmptan=np.tan(0.5*_rad*lat_n[I]+0.25*_pi_1)
            jpiv[I]=(np.log(tmptan)-self._slim*_rad)/(_rad*self._dj) + 1.0
        else :
            jpiv=(lat_n-self._slim)/self._dj + 1.0

        # Returns floats, cast to int to use as index/pivot points
        return ipiv,jpiv

    def get_grid_point(self,i,j,shifti=0.,shiftj=0.) :
        #! Used when creating a grid
        #! Retrieves  lon/lat for grid index (i,j) at P points
        #lon_n=self._wlim+(i-1)self._di+shifti
        #lat_n=self._slim+(j-1)*self._dj+shiftj
        lon_n=self._wlim+(i-1+shifti)*self._di
        lat_n=self._slim+(j-1+shiftj)*self._dj
        if self._mercator :
            lat_n=self._slim+(j-1+shiftj)*self._dm
            lat_n=(2.*np.arctan(np.exp((lat_n*_rad)))-_pi_1*.5)*_deg
        return self.newtoold(lat_n,lon_n)

    def ll2gind(self,lat_o,lon_o) :
        """ Returns grid index (floats) of specified lat and lon coordinates on the grid"""
        lat_n,lon_n = self.oldtonew(lat_o,lon_o)
        return self.pivotp(lat_n,lon_n)

    def gind2ll(self,i,j,shifti=0.,shiftj=0.):
        """ Returns lat and lon for grid index i,j """
        return self.get_grid_point(i,j,shifti=shifti,shiftj=shiftj)


