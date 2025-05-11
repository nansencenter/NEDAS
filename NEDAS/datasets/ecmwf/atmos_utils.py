import numpy
import datetime

_stefanb=5.67e-8
_s0=1365.                 # w/m^2  solar constant
_absh2o=0.09              # ---    absorption of water and ozone
_airdns=1.2

def lwrad_berliand(tair,e,cc) :
   # downwelling longwave radiation
   #tair : air temperature [K]
   #e    : near surface vapor pressure [Pa]
   #cc   : cloud cover (0-1)

   # Below formula assumes pressure in mBar
   e_mbar = e * 0.01
   emiss=0.97
   # Clear sky downwelling longwave flux from Eimova(1961)
   strd = emiss*_stefanb * tair**4 * (0.39-0.05*numpy.sqrt(e_mbar))

   # Cloud correction by Berliand (1952)
   strd = strd * (1. - 0.6823 * cc*cc)

   return strd


def strd_bignami(tair,e,cc) :
   # downwelling longwave radiation
   #tair : air temperature [K]
   #e    : near surface vapor pressure [Pa]
   #cc   : cloud cover (0-1)

   # Below formula assumes pressure in mBar (1hPa=1 mbar; 1Pas=0.01mbar)
   e_mbar = e * 0.01
   emiss=0.97
   # Clear sky downwelling longwave flux from Bignami(1995)
   strd = _stefanb * tair**4 * (0.653+0.00535*e_mbar)

   # Cloud correction by Berliand (1952)
   strd = strd * (1. + 0.1762 * cc*cc)

   return strd


def lwrad_budyko(lat,tair,e,cc) :
   # downwelling longwave radiation
   #tair : air temperature [K]
   #e    : near surface vapor pressure [Pa]
   #cc   : cloud cover (0-1)

   et=611.*10.**(7.5*(tair-273.16)/(tair-35.86))
#   print numpy.max(tair),numpy.max(et),numpy.max(e)
#   exit(0)
   # Below formula assumes pressure in mBar
   emiss=0.97
   deg2rad = numpy.pi/180.
   # Clear sky downwelling longwave flux from Budyko(1961)
   chi = 0.5+0.246*numpy.abs(lat*deg2rad)
   cc_cliped=numpy.minimum(1.0,numpy.maximum(cc,0.))
   term1=(0.254-4.95e-5*et)*(1.0-chi*cc_cliped**(1.2) )
   strd = emiss*_stefanb * tair**4 * (term1)


   return strd


#MOSTAFA: END


#http://www.nersc.no/biblio/formulation-air-sea-fluxes-esop2-version-micom
#http://regclim.met.no/rapport_4/presentation16/presentation16.htm
#def qlwd(tair,plat,cc,td)
#   fqlw  =emiss*stefanb*tair**3
#   fqlwcc=1.-(.5+.246*abs(plat(i,j)*radian))*cc**1.2
#c
#      fqlwi1=fqlw*tair*((.254-4.95e-5*vpair_i)*fqlwcc-4.)
#      fqlwi2=fqlw*4.
#c
#      fqlww1=fqlw*tair*((.254-4.95e-5*vpair_w)*fqlwcc-4.)
#      fqlww2=fqlwi2

def strd_efimova_jacobs(tair,e,cc) :
   #tair : air temperature [K]
   #e    : near surface vapor pressure [Pa]
   #cc   : cloud cover (0-1)

   # Below formula assumes pressure in mBar
   e_mbar = e * 0.01

   # Clear sky downwelling longwave flux from Eimova(1961)
   strd = _stefanb * tair**4 * (0.746+0.0066*e_mbar) 

   # Cloud correction by Jacobs(1978)
   strd = strd * (1. + 0.26 * cc) 

   return strd

def strd_maykut_jacobs(tair,e,cc) :
   #tair : air temperature [K]
   #cc   : cloud cover (0-1)

   # Below formula assumes pressure in mBar
   e_mbar = e * 0.01

   # Clear sky downwelling longwave flux from Maykut and Church (1973). 
   strd = _stefanb * tair**4 * 0.7855

   # Cloud correction by Jacobs(1978)
   strd = strd * (1. + 0.26 * cc) 

   return strd

def windstress(uwind,vwind) :
   karalight=True
   ws=numpy.sqrt(uwind**2+vwind**2)
   if karalight :
      wndfac=numpy.maximum(2.5,numpy.minimum(32.5,ws))
      cd_new = 1.0E-3*(.692 + .0710*wndfac - .000700*wndfac**2)
   # else :
   #    wndfac=(1.+numpy.sign(1.,ws-11.))*.5
   #    cd_new=(0.49+0.065*ws)*1.0e-3*wndfac+cd*(1.-wndfac)
   wfact=ws*_airdns*cd_new
   taux = uwind*wfact
   tauy = vwind*wfact
   ustar = numpy.sqrt((taux**2+tauy**2)*1e-3)
   return taux, tauy

def vapmix(e,p) :
   # Input is :
   # e = vapour pressure (partial pressure of vapour)
   # p = air pressure
   vapmix = 0.622 * e / (p-e)
   return vapmix

def satvap(t) :
   # This function calculates the saturation vapour pressure
   # [Pa] from the temperature [deg K].
   # Modified: Anita Jacob, June '97
   #
   # Input: t: temperature [deg K]
   # Output: satvap: saturation vapour pressure at temp. t
   #
   # es(T) = C1 * exp(C3*(T - T0)/(T - C4)) from ECMWF manual
   #data c1/610.78/,t00/273.16/
   c1=610.78
   t00=273.16

   #if (t < t00) then
   #   c3 = 21.875
   #   c4 = 7.66
   #else
   #   c3 = 17.269
   #   c4 = 35.86
   #endif
   #KAL !!! c3 = numpy.where(t < t00,21.875,7.66)
   #KAL !!! c4 = numpy.where(t < t00,17.269,35.86)

   # Old hycom
   #c3 = numpy.where(t < t00, 21.875,17.269)
   #c4 = numpy.where(t < t00,  7.66, 35.86)

   # From newest IFS (CY41R2)
   c3 = numpy.where(t < t00, 22.587,17.502)
   c4 = numpy.where(t < t00, -0.7  ,32.19)

   aa = c3 * (t - t00)
   bb = t - c4
   cc=aa/bb

   #if (cc < -20.0) then
   #   satvap=0.0
   #else
   #   satvap = c1 * exp(aa/bb)
   satvap=numpy.where(cc < -20.0, 0.0, c1 * numpy.exp(aa/bb))
   return satvap

def  relhumid(sva,svd,msl) :
   # This routine calculates the relative humidity by the
   # dew point temperature and the mean sea level pressure.
   # Modified: Anita Jacob, June '97
   # Input:
   #    sva: saturatn vapour press at air temp [K]
   #    svd: saturatn vapour press at dew pt temp [K]
   #    msl: pressure at mean sea level [Pa]
   # Output: 
   #   relhumid: Relative Humidity

   # We use the Tetens formula:
   # es(T) = C1 * exp(C3*(T - T0)/(T - C4)) from ECMWF manual
   #              es(Tdew)        p - es(Tair)
   # RH = 100 *  -----------   *  ------------
   #             p - es(tdew)       es(Tair)
   aaa=msl - svd
   aaa = svd/aaa
   bbb = (msl - sva)/sva
   relhumid = 100. * aaa * bbb
   return relhumid

def qsw_allsky_rosato(srad_top,cosz,cosz_noon,cc) :
   # Follows Rosato and Miyakoda[1988]
   # srad = cloud-top incident radiation
   # cosz = cosine of solar zenith angle

   # direct component
   sdir=srad_top*0.7**(1./(cosz+1e-2))     #direct radiation component
   sdif=((1.-_absh2o)*srad_top-sdir)*.5        #diffusive radiation component

   # Solar altitude
   altdeg=numpy.maximum(0.,numpy.arcsin(cosz_noon))*180./numpy.pi #solar noon altitude in degrees

   cfac=(1.-0.62*cc+0.0019*altdeg)               #cloudiness correction by Reed(1977)
   ssurf=(sdir+sdif)*cfac

   return ssurf









def qsw0(qswtime,daysinyear,cc,plat,plon) :
   #
   # --- -------------------------------------------------------------------
   # --- compute 24 hrs mean solar irrradiance at the marine surface layer
   # --- (unit: w/m^2)
   # --- -------------------------------------------------------------------
   #
   # --- Average number of days in year over a 400-year cycle (Gregorian Calendar)
   daysinyear400=365.2425
   #c --- set various quantities
   pi2=8.*numpy.arctan(1.)          #        2 times pi
   deg=360./pi2             #        convert from radians to degrees
   rad=pi2/360.             #        convert from degrees to radians
   eepsil=1.e-9             #        small number
   ifrac=24                 #        split each 12 hrs day into ifrac parts
   fraci=1./ifrac           #        1 over ifrac
   absh2o=0.09              # ---    absorption of water and ozone
   s0=1365.                 # w/m^2  solar constant
   radian=rad
#c
#c --- -------------------------------------------------------------------
#c --- compute 24 hrs mean solar radiation at the marine surface layer 
#c --- -------------------------------------------------------------------
#C --- KAL: TODO - adhere to hycom time setup
   day=numpy.mod(qswtime,daysinyear)    #0 < day < 364
   day=numpy.floor(day)
#c
   dangle=pi2*day/float(daysinyear)   #day-number-angle, in radians 
   if day<0. or day>daysinyear+1 :
      print('qsw0: Error in day for day angle')
      print('Day angle is ',day,daysinyear,qswtime)
      raise NameError("test")
      
# --- compute astronomic quantities -- 
   decli=.006918+.070257*numpy.sin(dangle)   -.399912*numpy.cos(dangle)      \
                +.000907*numpy.sin(2.*dangle)-.006758*numpy.cos(2.*dangle)   \
                +.001480*numpy.sin(3.*dangle)-.002697*numpy.cos(3.*dangle)

   sundv=1.00011+.001280*numpy.sin(dangle)   +.034221*numpy.cos(dangle)      \
                +.000077*numpy.sin(2.*dangle)+.000719*numpy.cos(2.*dangle)

# --- compute astronomic quantities

   sin2=numpy.sin(plat*radian)*numpy.sin(decli)
   cos2=numpy.cos(plat*radian)*numpy.cos(decli)
#
# --- split each day into ifrac parts, and compute the solar radiance for 
# --- each part. by assuming symmetry of the irradiance about noon, it
# --- is sufficient to compute the irradiance for the first 12 hrs of
# --- the (24 hrs) day (mean for the first 12 hrs equals then the mean
# --- for the last 12 hrs)
#
# --- TODO - This routine can also return daily varying solar heat flux
   scosz=0.
   stot=0.
   for npart in range(1,25) :
      bioday=day+(npart-.5)*fraci*.5
      biohr=bioday*86400.                #hour of day in seconds
      biohr=numpy.mod(biohr+43200.,86400.)    #hour of day;  biohr=0  at noon
      hangle=pi2*biohr/86400.            #hour angle, in radians
#
      cosz=numpy.maximum(0.,sin2+cos2*numpy.cos(hangle)) #cosine of the zenith angle
      scosz=scosz+cosz                     #  ..accumulated..
      srad =s0*sundv*cosz                  #extraterrestrial radiation
#
#         sdir=srad*0.7**(1./(cosz+eepsil))    #direct radiation component
#         sdir=srad * exp(-0.356674943938732447/(cosz+eepsil))         
# ---    KAL prevent underflow - .7^100 = 3x10^-16 
      sdir=srad*0.7**(numpy.minimum(100.,1./(cosz+eepsil)))    #direct radiation component
#
      sdif=((1.-absh2o)*srad-sdir)*.5               #diffusive radiation component
      altdeg=numpy.maximum(0.,numpy.arcsin(numpy.minimum(1.0,sin2+cos2)))*deg #solar noon altitude in degrees
      cfac=(1.-0.62*cc+0.0019*altdeg)               #cloudiness correction 
      ssurf=(sdir+sdif)*cfac
      stot=stot+ssurf

#     enddo
   scosz=scosz*fraci               #24-hrs mean of  cosz
   radfl0=stot*fraci               #24-hrs mean shortw rad in w/m^2
#
# --  Original formula was wrong ...
#     !cawdir(i,j)=1.-numpy.maximum(0.15,0.05/(scosz+0.15)) 
#     !cawdir(i,j)=1.-numpy.maximum(0.03,0.05/(scosz+0.15))  !Correction   - Mats
   cawdir=1.-numpy.minimum(0.15,0.05/(scosz+0.15))   #Correction 2 - KAL
#     enddo
#     enddo
#     enddo
#$OMP END PARALLEL DO
#
#     end subroutine qsw0

   return radfl0,cawdir

