import os
import numpy as np
from datetime import datetime, timedelta, timezone
from ..time_format import forday, dayfor

def bool_str(value):
    if value:
        vstr = 'T'
    else:
        vstr = 'F'
    return vstr

def blkdat(m):
    ##generate blkdat.input based on model object m
    nmlstr =  f"ECWMF forcing; flx-s14w; LWcorr; precip+2mm; SSSrlx; FCT2 tsadvc.; 0-tracer.\n"
    nmlstr += f"Sigma0; GDEM3 init; KPP mixed layer; SeaWiFS mon KPAR; nested in ATLd0.08 2.6;\n"
    nmlstr += f"S-Z(15-11): dp00/f/x/i=3m/1.125/12m/1m; ds=1m/1.125/4m; src_2.2.12;\n"
    nmlstr += f"12345678901234567890123456789012345678901234567890123456789012345678901234567890\n"
    nmlstr += f" {m.V[0]}{m.V[2]}       'iversn' = hycom version number x10\n"
    nmlstr += f" {m.E}      'iexpt ' = experiment number x10\n"
    nmlstr += f"{m.idm:4d}      'idm   ' = longitudinal array size\n"
    nmlstr += f"{m.jdm:4d}      'jdm   ' = latitudinal  array size\n"
    nmlstr += f"{m.itest:4d}      'itest ' = grid point where detailed diagnostics are desired\n"
    nmlstr += f"{m.jtest:4d}      'jtest ' = grid point where detailed diagnostics are desired\n"
    nmlstr += f"{m.kdm:4d}      'kdm   ' = number of layers\n"
    nmlstr += f"{m.nhybrd:4d}      'nhybrd' = number of hybrid levels (0=all isopycnal)\n"
    nmlstr += f"{m.nsigma:4d}      'nsigma' = number of sigma  levels (nhybrd-nsigma z-levels)\n"
    nmlstr += f"{str(m.dp00):^8s}  'dp00  ' = deep    z-level spacing minimum thickness (m)\n"
    nmlstr += f"{str(m.dp00x):^8s}  'dp00x ' = deep    z-level spacing maximum thickness (m)\n"
    nmlstr += f"{str(m.dp00f):^8s}  'dp00f ' = deep    z-level spacing stretching factor (1.0=const.space)\n"
    nmlstr += f"{str(m.ds00):^8s}  'ds00  ' = shallow z-level spacing minimum thickness (m)\n"
    nmlstr += f"{str(m.ds00x):^8s}  'ds00x ' = shallow z-level spacing maximum thickness (m)\n"
    nmlstr += f"{str(m.ds00f):^8s}  'ds00f ' = shallow z-level spacing stretching factor (1.0=const.space)\n"
    nmlstr += f"{str(m.dp00i):^8s}  'dp00i ' = deep iso-pycnal spacing minimum thickness (m)\n"
    nmlstr += f"{str(m.isotop):^8s}  'isotop' = shallowest depth for isopycnal layers     (m, <0 from file)\n"
    nmlstr += f"{str(m.saln0):^8s}  'saln0 ' = initial salinity value (psu), only used for iniflg<2\n"
    nmlstr += f"{str(m.locsig):^8s}  'locsig' = locally-referenced pot. density for stability (0=F,1=T)\n"
    nmlstr += f"{str(m.kapref):^8s}  'kapref' = thermobaric ref. state (-1=input,0=none,1,2,3=constant)\n"
    nmlstr += f"{str(m.thflag):^8s}  'thflag' = reference pressure flag (0=Sigma-0, 2=Sigma-2)\n"
    nmlstr += f"{str(m.thbase):^8s}  'thbase' = reference density (sigma units)\n"
    nmlstr += f"{str(m.vsigma):^8s}  'vsigma' = spacially varying isopycnal target densities (0=F,1=T)\n"
    for k in range(m.kdm):
        nmlstr += f"{str(m.sigma[k]):^8s}  'sigma ' = layer  {k+1}  density (sigma units)\n"
    nmlstr += f"{str(m.iniflg):^8s}  'iniflg' = initial state flag (0=levl, 1=zonl, 2=clim)\n"
    nmlstr += f"{str(m.jerlv0):^8s}  'jerlv0' = initial jerlov water type (1 to 5; 0 to use KPAR)\n"
    nmlstr += f"{str(m.yrflag):^8s}  'yrflag' = days in year flag   (0=360,  1=366,  2=366J1, 3=actual)\n"
    nmlstr += f"{str(m.sshflg):^8s}  'sshflg' = diagnostic SSH flag (0=SSH,1=SSH&stericSSH)\n"
    nmlstr += f"{str(m.dsurfq):^8s}  'dsurfq' = number of days between model diagnostics at the surface\n"
    nmlstr += f"{str(m.diagfq):^8s}  'diagfq' = number of days between model diagnostics\n"
    nmlstr += f"{str(m.proffq):^8s}  'proffq' = number of days between model diagnostics at selected locs\n"
    nmlstr += f"{str(m.tilefq):^8s}  'tilefq' = number of days between model diagnostics on selected tiles\n"
    nmlstr += f"{str(m.meanfq):^8s}  'meanfq' = number of days between model diagnostics (time averaged)\n"
    nmlstr += f"{str(m.rstrfq):^8s}  'rstrfq' = number of days between model restart output\n"
    nmlstr += f"{str(m.bnstfq):^8s}  'bnstfq' = number of days between baro nesting archive input\n"
    nmlstr += f"{str(m.nestfq):^8s}  'nestfq' = number of days between 3-d  nesting archive input\n"
    nmlstr += f"{str(m.cplifq):^8s}  'cplifq' = number of days (or time steps) between sea ice coupling\n"
    nmlstr += f"{str(m.baclin):^8s}  'baclin' = baroclinic time step (seconds), int. divisor of 86400\n"
    nmlstr += f"{str(m.batrop):^8s}  'batrop' = barotropic time step (seconds), int. div. of baclin/2\n"
    nmlstr += f"{str(m.incflg):^8s}  'incflg' = incremental update flag (0=no, 1=yes, 2=full-velocity)\n"
    nmlstr += f"{str(m.incstp):^8s}  'incstp' = no. timesteps for full update (1=direct insertion)\n"
    nmlstr += f"{str(m.incupf):^8s}  'incupf' = number of days of incremental updating input\n"
    nmlstr += f"{str(m.ra2fac):^8s}  'ra2fac' = weight for Robert-Asselin time filter\n"
    nmlstr += f"{str(m.wbaro ):^8s}  'wbaro ' = barotropic time smoothing weight\n"
    nmlstr += f"{str(m.btrlfr):^8s}  'btrlfr' = leapfrog barotropic time step (0=F,1=T)\n"
    nmlstr += f"{str(m.btrmas):^8s}  'btrmas' = barotropic is mass conserving (0=F,1=T)\n"
    nmlstr += f"{str(m.hybraf):^8s}  'hybraf' = HYBGEN: Robert-Asselin flag   (0=F,1=T)\n"
    nmlstr += f"{str(m.hybrlx):^8s}  'hybrlx' = HYBGEN: inverse relaxation coefficient (time steps)\n"
    nmlstr += f"{str(m.hybiso):^8s}  'hybiso' = HYBGEN: Use PCM if layer is within hybiso of target density\n"
    nmlstr += f"{str(m.hybmap):^8s}  'hybmap' = hybrid   remapper  flag (0=PCM, 1=PLM,  2=PPM)\n"
    nmlstr += f"{str(m.hybflg):^8s}  'hybflg' = hybrid   generator flag (0=T&S, 1=th&S, 2=th&T)\n"
    nmlstr += f"{str(m.advflg):^8s}  'advflg' = thermal  advection flag (0=T&S, 1=th&S, 2=th&T)\n"
    nmlstr += f"{str(m.advtyp):^8s}  'advtyp' = scalar   advection type (0=PCM,1=MPDATA,2=FCT2,4=FCT4)\n"
    nmlstr += f"{str(m.momtyp):^8s}  'momtyp' = momentum advection type (2=2nd order, 4=4th order)\n"
    nmlstr += f"{str(m.slip  ):^8s}  'slip  ' = +1 for free-slip, -1 for non-slip boundary conditions\n"
    nmlstr += f"{str(m.visco2):^8s}  'visco2' = deformation-dependent Laplacian  viscosity factor\n"
    nmlstr += f"{str(m.visco4):^8s}  'visco4' = deformation-dependent biharmonic viscosity factor\n"
    nmlstr += f"{str(m.facdf4):^8s}  'facdf4' =       speed-dependent biharmonic viscosity factor\n"
    nmlstr += f"{str(m.veldf2):^8s}  'veldf2' = diffusion velocity (m/s) for Laplacian  momentum dissip.\n"
    nmlstr += f"{str(m.veldf4):^8s}  'veldf4' = diffusion velocity (m/s) for biharmonic momentum dissip.\n"
    nmlstr += f"{str(m.thkdf2):^8s}  'thkdf2' = diffusion velocity (m/s) for Laplacian  thickness diffus.\n"
    nmlstr += f"{str(m.thkdf4):^8s}  'thkdf4' = diffusion velocity (m/s) for biharmonic thickness diffus.\n"
    nmlstr += f"{str(m.temdf2):^8s}  'temdf2' = diffusion velocity (m/s) for Laplacian  temp/saln diffus.\n"
    nmlstr += f"{str(m.temdfc):^8s}  'temdfc' = temp diffusion conservation (0.0,1.0 all dens,temp resp.)\n"
    nmlstr += f"{str(m.vertmx):^8s}  'vertmx' = diffusion velocity (m/s) for momentum at MICOM M.L.base\n"
    nmlstr += f"{str(m.cbar  ):^8s}  'cbar  ' = rms flow speed     (m/s) for linear bottom friction\n"
    nmlstr += f"{str(m.cb    ):^8s}  'cb    ' = coefficient of quadratic bottom friction\n"
    nmlstr += f"{str(m.drglim):^8s}  'drglim' = limiter for explicit friction (1.0 none, 0.0 implicit)\n"
    nmlstr += f"{str(m.thkbot):^8s}  'thkbot' = thickness of bottom boundary layer (m)\n"
    nmlstr += f"{str(m.sigjmp):^8s}  'sigjmp' = minimum density jump across interfaces  (kg/m**3)\n"
    nmlstr += f"{str(m.tmljmp):^8s}  'tmljmp' = equivalent temperature jump across mixed-layer (degC)\n"
    nmlstr += f"{str(m.thkmls):^8s}  'thkmls' = reference mixed-layer thickness for SSS relaxation (m)\n"
    nmlstr += f"{str(m.thkmlt):^8s}  'thkmlt' = reference mixed-layer thickness for SST relaxation (m)\n"
    nmlstr += f"{str(m.thkriv):^8s}  'thkriv' = nominal thickness of river inflow (m)\n"
    nmlstr += f"{str(m.thkcdw):^8s}  'thkcdw' = thickness for near-surface currents in ice-ocean stress (m)\n"
    nmlstr += f"{str(m.thkfrz):^8s}  'thkfrz' = maximum thickness of near-surface freezing zone (m)\n"
    nmlstr += f"{str(m.iceflg):^8s}  'iceflg' = sea ice model flag (0=none,1=energy loan,2=coupled/esmf)\n"
    nmlstr += f"{str(m.tfrz_0):^8s}  'tfrz_0' = ENLN: ice melting point (degC) at S=0psu\n"
    nmlstr += f"{str(m.tfrz_s):^8s}  'tfrz_s' = ENLN: gradient of ice melting point (degC/psu)\n"
    nmlstr += f"{str(m.frzifq):^8s}  'frzifq' = e-folding time scale back to tfrz (days or -ve time steps)\n"
    nmlstr += f"{str(m.ticegr):^8s}  'ticegr' = ENLN: temp. grad. inside ice (deg/m); =0 use surtmp\n"
    nmlstr += f"{str(m.hicemn):^8s}  'hicemn' = ENLN: minimum ice thickness (m)\n"
    nmlstr += f"{str(m.hicemx):^8s}  'hicemx' = ENLN: maximum ice thickness (m)\n"
    nmlstr += f"{str(m.ishelf):^8s}  'ishelf' = ice shelf flag    (0=none,1=ice shelf over ocean)\n"
    nmlstr += f"{str(m.ntracr):^8s}  'ntracr' = number of tracers (0=none,negative to initialize)\n"
    nmlstr += f"{str(m.trcflg):^8s}  'trcflg' = tracer flags      (one digit per tr, most sig. replicated)\n"
    nmlstr += f"{str(m.tsofrq):^8s}  'tsofrq' = number of time steps between anti-drift offset calcs\n"
    nmlstr += f"{str(m.tofset):^8s}  'tofset' = temperature anti-drift offset (degC/century)\n"
    nmlstr += f"{str(m.sofset):^8s}  'sofset' = salnity     anti-drift offset  (psu/century)\n"
    nmlstr += f"{str(m.mlflag):^8s}  'mlflag' = mixed layer flag  (0=none,1=KPP,2-3=KT,4=PWP,5=MY,6=GISS)\n"
    nmlstr += f"{str(m.pensol):^8s}  'pensol' = KT:      activate penetrating solar rad.   (0=F,1=T)\n"
    nmlstr += f"{str(m.dtrate):^8s}  'dtrate' = KT:      maximum permitted m.l. detrainment rate  (m/day)\n"
    nmlstr += f"{str(m.thkmin):^8s}  'thkmin' = KT/PWP:  minimum mixed-layer thickness (m)\n"
    nmlstr += f"{str(m.dypflg):^8s}  'dypflg' = KT/PWP:  diapycnal mixing flag (0=none, 1=KPP, 2=explicit)\n"
    nmlstr += f"{str(m.mixfrq):^8s}  'mixfrq' = KT/PWP:  number of time steps between diapycnal mix calcs\n"
    nmlstr += f"{str(m.diapyc):^8s}  'diapyc' = KT/PWP:  diapycnal diffusivity x buoyancy freq. (m**2/s**2)\n"
    nmlstr += f"{str(m.rigr  ):^8s}  'rigr  ' = PWP:     critical gradient richardson number\n"
    nmlstr += f"{str(m.ribc  ):^8s}  'ribc  ' = PWP:     critical bulk     richardson number\n"
    nmlstr += f"{str(m.rinfty):^8s}  'rinfty' = KPP:     maximum  gradient richardson number (shear inst.)\n"
    nmlstr += f"{str(m.ricr  ):^8s}  'ricr  ' = KPP:     critical bulk     richardson number\n"
    nmlstr += f"{str(m.bldmin):^8s}  'bldmin' = KPP:     minimum surface boundary layer thickness (m)\n"
    nmlstr += f"{str(m.bldmax):^8s}  'bldmax' = K-PROF:  maximum surface boundary layer thickness (m)\n"
    nmlstr += f"{str(m.cekman):^8s}  'cekman' = KPP/KT:  scale factor for Ekman depth\n"
    nmlstr += f"{str(m.cmonob):^8s}  'cmonob' = KPP:     scale factor for Monin-Obukov depth\n"
    nmlstr += f"{str(m.bblkpp):^8s}  'bblkpp' = KPP:     activate bottom boundary layer    (0=F,1=T)\n"
    nmlstr += f"{str(m.shinst):^8s}  'shinst' = KPP:     activate shear instability mixing (0=F,1=T)\n"
    nmlstr += f"{str(m.dbdiff):^8s}  'dbdiff' = KPP:     activate double diffusion  mixing (0=F,1=T)\n"
    nmlstr += f"{str(m.nonloc):^8s}  'nonloc' = KPP:     activate nonlocal b. layer mixing (0=F,1=T)\n"
    nmlstr += f"{str(m.botdiw):^8s}  'botdiw' = K_PROF:    activate bot.enhan.int.wav mixing (0=F,1=T)\n"
    nmlstr += f"{str(m.difout):^8s}  'difout' = K-PROF:  output visc/diff coffs in archive (0=F,1=T)\n"
    nmlstr += f"{str(m.difsmo):^8s}  'difsmo' = K-PROF:  number of layers with horiz smooth diff coeffs\n"
    nmlstr += f"{str(m.difm0):^8s}  'difm0 ' = KPP:     max viscosity   due to shear instability (m**2/s)\n"
    nmlstr += f"{str(m.difs0):^8s}  'difs0 ' = KPP:     max diffusivity due to shear instability (m**2/s)\n"
    nmlstr += f"{str(m.difmiw):^8s}  'difmiw' = KPP:     background/internal wave viscosity       (m**2/s)\n"
    nmlstr += f"{str(m.difsiw):^8s}  'difsiw' = KPP:     background/internal wave diffusivity     (m**2/s)\n"
    nmlstr += f"{str(m.dsfmax):^8s}  'dsfmax' = KPP:     salt fingering diffusivity factor        (m**2/s)\n"
    nmlstr += f"{str(m.rrho0):^8s}  'rrho0 ' = KPP:     salt fingering rp=(alpha*delT)/(beta*delS)\n"
    nmlstr += f"{str(m.cs):^8s}  'cs    ' = KPP:     value for nonlocal flux term\n"
    nmlstr += f"{str(m.cstar):^8s}  'cstar ' = KPP:     value for nonlocal flux term\n"
    nmlstr += f"{str(m.cv):^8s}  'cv    ' = KPP:     buoyancy frequency ratio (0.0 to use a fn. of N)\n"
    nmlstr += f"{str(m.c11):^8s}  'c11   ' = KPP:     value for turb velocity scale\n"
    nmlstr += f"{str(m.hblflg):^8s}  'hblflg' = KPP:     b. layer interp. flag (0=const.,1=linear,2=quad.)\n"
    nmlstr += f"{str(m.niter):^8s}  'niter ' = KPP:     iterations for semi-implicit soln. (2 recomended)\n"
    nmlstr += f"{str(m.langmr):^8s}  'langmr' = KPP:     activate Langmuir turb. factor    (0=F,1=T)\n"
    nmlstr += f"{str(m.fltflg):^8s}  'fltflg' = FLOATS: synthetic float flag (0=no; 1=yes)\n"
    nmlstr += f"{str(m.nfladv):^8s}  'nfladv' = FLOATS: advect every nfladv bacl. time steps (even, >=4)\n"
    nmlstr += f"{str(m.nflsam):^8s}  'nflsam' = FLOATS: output (0=every nfladv steps; >0=no. of days)\n"
    nmlstr += f"{str(m.intpfl):^8s}  'intpfl' = FLOATS: horiz. interp. (0=2nd order+n.n.; 1=n.n. only)\n"
    nmlstr += f"{str(m.iturbv):^8s}  'iturbv' = FLOATS: add horiz. turb. advection velocity (0=no; 1=yes)\n"
    nmlstr += f"{str(m.ismpfl):^8s}  'ismpfl' = FLOATS: sample water properties at float (0=no; 1=yes)\n"
    nmlstr += f"{str(m.tbvar):^8s}  'tbvar ' = FLOATS: horizontal turb. vel. variance scale (m**2/s**2)\n"
    nmlstr += f"{str(m.tdecri):^8s}  'tdecri' = FLOATS: inverse decorrelation time scale (1/day)\n"
    nmlstr += f"{str(m.lbflag):^8s}  'lbflag' = lateral barotropic bndy flag (0=none, 1=port, 2=input)\n"
    nmlstr += f"{str(m.tidflg):^8s}  'tidflg' = TIDES: tidal forcing flag    (0=none,1=open-bdy,2=bdy&body)\n"
    nmlstr += f"{str(m.tidein):^8s}  'tidein' = TIDES: tide field input flag (0=no;1=yes;2=sal)\n"
    nmlstr += f"{str(m.tidcon):^8s}  'tidcon' = TIDES: 1 digit per (Q1K2P1N2O1K1S2M2), 0=off,1=on\n"
    nmlstr += f"{str(m.tidsal):^8s}  'tidsal' = TIDES: scalar self attraction and loading factor (<0: file)\n"
    nmlstr += f"{str(m.tiddrg):^8s}  'tiddrg' = TIDES: tidal drag flag (0=no;1=scalar;2=tensor)\n"
    nmlstr += f"{str(m.thkdrg):^8s}  'thkdrg' = thickness of bottom boundary layer for tidal drag (m)\n"
    nmlstr += f"{str(m.drgscl):^8s}  'drgscl' = scale factor for tidal drag   (0.0 for no tidal drag)\n"
    nmlstr += f"{str(m.tidgen):^8s}  'tidgen' = TIDES: generic time (0=F,1=T)\n"
    nmlstr += f"{str(m.tidrmp):^8s}  'tidrmp' = TIDES:            ramp time (days)\n"
    nmlstr += f"{str(m.tid_t0):^8s}  'tid_t0' = TIDES: origin for ramp time (model day)\n"
    nmlstr += f"{str(m.clmflg):^8s}  'clmflg' = climatology frequency flag   (6=bimonthly, 12=monthly)\n"
    nmlstr += f"{str(m.wndflg):^8s}  'wndflg' = wind stress input flag (0=none,1=u/v-grid,2,3=p-grid,4,5=wnd10m)\n"
    nmlstr += f"{str(m.ustflg):^8s}  'ustflg' = ustar forcing     flag        (3=input,1,2=wndspd,4=stress)\n"
    nmlstr += f"{str(m.flxflg):^8s}  'flxflg' = thermal forcing   flag (0=none,3=net-flux,1,2,4-6=sst-based)\n"
    nmlstr += f"{str(m.empflg):^8s}  'empflg' = E-P     forcing   flag (0=none,3=net_E-P, 1,2,4-6=sst-based_E)\n"
    nmlstr += f"{str(m.dswflg):^8s}  'dswflg' = diurnal shortwave flag (0=none,1=daily to diurnal corr.)\n"
    nmlstr += f"{str(m.albflg):^8s}  'albflg' = ocean albedo      flag (0=none,1=const,2=L&Y)\n"
    nmlstr += f"{str(m.sssflg):^8s}  'sssflg' = SSS relaxation flag (0=none,1=clim)\n"
    nmlstr += f"{str(m.lwflag):^8s}  'lwflag' = longwave (SST) flag (0=none,1=clim,2=atmos)\n"
    nmlstr += f"{str(m.sstflg):^8s}  'sstflg' = SST relaxation flag (0=none,1=clim,2=atmos,3=observed)\n"
    nmlstr += f"{str(m.icmflg):^8s}  'icmflg' = ice mask       flag (0=none,1=clim,2=atmos,3=obs/coupled)\n"
    nmlstr += f"{str(m.prsbas):^8s}  'prsbas' = msl pressure is input field + prsbas (Pa)\n"
    nmlstr += f"{str(m.mslprf):^8s}  'mslprf' = msl pressure forcing flag            (0=F,1=T)\n"
    nmlstr += f"{str(m.stroff):^8s}  'stroff' = net stress offset flag               (0=F,1=T)\n"
    nmlstr += f"{str(m.flxoff):^8s}  'flxoff' = net flux offset flag                 (0=F,1=T)\n"
    nmlstr += f"{str(m.flxsmo):^8s}  'flxsmo' = smooth surface fluxes                (0=F,1=T)\n"
    nmlstr += f"{str(m.relax):^8s}  'relax ' = activate lateral boundary nudging    (0=F,1=T)\n"
    nmlstr += f"{str(m.trcrlx):^8s}  'trcrlx' = activate lat. bound. tracer nudging  (0=F,1=T)\n"
    nmlstr += f"{str(m.priver):^8s}  'priver' = rivers as a precipitation bogas      (0=F,1=T)\n"
    nmlstr += f"{str(m.epmass):^8s}  'epmass' = treat evap-precip as a mass exchange (0=F,1=T)\n"
    nmlstr += f"{str(m.nmrsti):^8s}  'nmrsti' = 'restart'  Restart filename. Dot and date appended. Empty: use old hycom name.\n"
    nmlstr += f"{str(m.nmrsto):^8s}  'nmrsto' = 'restart'  Restart filename. Dot and date appended. Empty: use old hycom name.\n"
    nmlstr += f"{str(m.nmarcs):^8s}  'nmarcs' = '' Surface archive filename. Dot and date appended. Empty: use old hycom name.\n"
    nmlstr += f"{str(m.nmarcv):^8s}  'nmarcv' = '' Full    archive filename. Dot and date appended. Empty: use old hycom name.\n"
    nmlstr += f"{str(m.nmarcm):^8s}  'nmarcm' = '' Mean    archive filename. Dot and date appended. Empty: use old hycom name.\n"
    nmlstr += f"{str(m.stdflg):^8s}  'stdflg' = STOKES: add Stokes Drift velocities to kinematics and dynamics (0=F,1=T)\n"
    nmlstr += f"{str(m.stdsur):^8s}  'stdsur' = STOKES: add Stokes Drift Surface Stresses  to dnamics (0=F,1=T)\n"
    nmlstr += f"{str(m.stdtau):^8s}  'stdtau' = Remov from the tot wind stress the part lost to wave : (0=F,1=T)\n"
    nmlstr += f"{str(m.stdwom):^8s}  'stdwom' = Add wave-to-ocean momentum flux due to wave breaking  : (0=F,1=T)\n"
    nmlstr += f"{str(m.stdbot):^8s}  'stdbot' = STOKES: add Stokes Waves Field bottom friction drag (0=F,1=T)\n"
    nmlstr += f"{str(m.stdarc):^8s}  'stdarc' = STOKES: Stokes Drift Velocities in Archive (0=F,1=T)\n"
    nmlstr += f"{str(m.altlng):^8s}  'altlng' = STOKES: Use alternative Lang def. from Kai et al (0=F,1=T)\n"
    nmlstr += f"{str(m.nsdzi):^8s}  'nsdzi' = STOKES: number of interfaces in Stokes Drift input\n"
    return nmlstr

def limits(m, time, forecast_period):
    next_time = time + forecast_period * timedelta(hours=1)
    fdtime = dayfor(m.yrflag, time.year, time.timetuple().tm_yday, time.hour)
    ldtime = dayfor(m.yrflag, next_time.year, next_time.timetuple().tm_yday, next_time.hour)
    nmlstr = "{:14.5f} {:14.5f}".format(fdtime, ldtime)
    return nmlstr

def ports(m):
    nmlstr =  f"{str(m.nports):^8s} 'nports   ' = number of ports\n"
    for i in range(m.nports):
        nmlstr += f"{str(m.kdport[i]):^8s} 'kdport   ' = port orientation (1=N, 2=S, 3=E, 4=W)\n"
        nmlstr += f"{str(m.ifport[i]):^8s} 'ifport   ' = first i-index\n"
        nmlstr += f"{str(m.ilport[i]):^8s} 'ilport   ' = last  i-index (=ifport for east/west port)\n"
        nmlstr += f"{str(m.jfport[i]):^8s} 'jfport   ' = first j-index\n"
        nmlstr += f"{str(m.jlport[i]):^8s} 'jlport   ' = last  j-index (=jfport for north/south port)\n"
    return nmlstr

def ice_in(m, time, forecast_period):
    year_init = 1958 ##time.year
    time_init = datetime(year_init, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    sec0 = (time - time_init) / timedelta(seconds=1)
    istep0 = int(np.floor(sec0 / m.cice_dt)) - 1
    npt = int(np.floor(forecast_period * 3600 / m.cice_dt)) + 1

    ##namelist input for cice model "ice_in"
    nmlstr  = f"&setup_nml\n"
    nmlstr += f"    days_per_year = 365\n"
    nmlstr += f"    use_leap_years = .true.\n"
    nmlstr += f"    year_init = {year_init}\n"
    nmlstr += f"    istep0 = {istep0}\n"
    nmlstr += f"    dt = {m.cice_dt}\n"
    nmlstr += f"    npt = {npt}\n"
    nmlstr += f"    ndtd = 1\n"
    nmlstr += f"    runtype = 'continue'\n"
    nmlstr += f"    ice_ic = './cice/iced_gx1_v5.nc'\n"
    nmlstr += f"    restart = .true.\n"
    nmlstr += f"    restart_ext = .false.\n"
    nmlstr += f"    use_restart_time = .true.\n"
    nmlstr += f"    restart_format = 'nc'\n"
    nmlstr += f"    lcdf64 = .false.\n"
    nmlstr += f"    restart_dir = './cice/'\n"
    nmlstr += f"    restart_file = 'iced'\n"
    nmlstr += f"    pointer_file = './cice/ice.restart_file'\n"
    nmlstr += f"    dumpfreq = 'd'\n"
    nmlstr += f"    dumpfreq_n = 60\n"
    nmlstr += f"    dump_last = .true.\n"
    nmlstr += f"    bfbflag = .false.\n"
    nmlstr += f"    diagfreq = 24\n"
    nmlstr += f"    diag_type = 'stdout'\n"
    nmlstr += f"    diag_file = 'ice_diag.d'\n"
    nmlstr += f"    print_global = .false.\n"
    nmlstr += f"    print_points = .false.\n"
    nmlstr += f"    latpnt(1:2) = 90.0, 75.0\n"
    nmlstr += f"    lonpnt(1:2) = 0.0, 145.0\n"
    nmlstr += f"    dbug = .false.\n"
    nmlstr += f"    histfreq = 'h', 'd', 'm', 'x', 'x'\n"
    nmlstr += f"    histfreq_n = 0, 1, 0, 1, 1\n"
    nmlstr += f"    hist_avg = .true.\n"
    nmlstr += f"    history_dir = './cice/'\n"
    nmlstr += f"    history_file = 'iceh'\n"
    nmlstr += f"    write_ic = .true.\n"
    nmlstr += f"    incond_dir = './cice/'\n"
    nmlstr += f"    incond_file = 'iceh_ic'\n"
    nmlstr += f"/\n"
    nmlstr += f"\n"
    nmlstr += f"&grid_nml\n"
    nmlstr += f"    grid_format = 'nc'\n"
    nmlstr += f"    grid_type = 'regional'\n"
    nmlstr += f"    grid_file = 'cice_grid.nc'\n"
    nmlstr += f"    kmt_file = 'cice_kmt.nc'\n"
    nmlstr += f"    gridcpl_file = 'unknown_gridcpl_file'\n"
    nmlstr += f"    kcatbound = 0\n"
    nmlstr += f"/\n"
    nmlstr += f"\n"
    nmlstr += f"&domain_nml\n"
    nmlstr += f"    nprocs = {m.nproc}\n"
    nmlstr += f"    processor_shape = 'square-pop'\n"
    nmlstr += f"    distribution_type = 'cartesian'\n"
    nmlstr += f"    distribution_wght = 'block'\n"
    nmlstr += f"    ew_boundary_type = 'open'\n"
    nmlstr += f"    ns_boundary_type = 'open'\n"
    nmlstr += f"    maskhalo_dyn = .false.\n"
    nmlstr += f"    maskhalo_remap = .false.\n"
    nmlstr += f"    maskhalo_bound = .false.\n"
    nmlstr += f"/\n"
    nmlstr += f"\n"
    nmlstr += f"&tracer_nml\n"
    nmlstr += f"    tr_iage = .true.\n"
    nmlstr += f"    restart_age = .false.\n"
    nmlstr += f"    tr_fy = .true.\n"
    nmlstr += f"    restart_fy = .false.\n"
    nmlstr += f"    tr_lvl = .true.\n"
    nmlstr += f"    restart_lvl = .false.\n"
    nmlstr += f"    tr_pond_cesm = .false.\n"
    nmlstr += f"    restart_pond_cesm = .false.\n"
    nmlstr += f"    tr_pond_topo = .false.\n"
    nmlstr += f"    restart_pond_topo = .false.\n"
    nmlstr += f"    tr_pond_lvl = .true.\n"
    nmlstr += f"    restart_pond_lvl = .false.\n"
    nmlstr += f"    tr_aero = .false.\n"
    nmlstr += f"    restart_aero = .false.\n"
    nmlstr += f"/\n"
    nmlstr += f"\n"
    nmlstr += f"&thermo_nml\n"
    nmlstr += f"    kitd = 1\n"
    nmlstr += f"    ktherm = 1\n"
    nmlstr += f"    conduct = 'bubbly'\n"
    nmlstr += f"    a_rapid_mode = 0.0005\n"
    nmlstr += f"    rac_rapid_mode = 10.0\n"
    nmlstr += f"    aspect_rapid_mode = 1.0\n"
    nmlstr += f"    dsdt_slow_mode = -5e-08\n"
    nmlstr += f"    phi_c_slow_mode = 0.05\n"
    nmlstr += f"    phi_i_mushy = 0.85\n"
    nmlstr += f"/\n"
    nmlstr += f"\n"
    nmlstr += f"&dynamics_nml\n"
    nmlstr += f"    kdyn = 1\n"
    nmlstr += f"    ndte = 120\n"
    nmlstr += f"    revised_evp = .false.\n"
    nmlstr += f"    advection = 'remap'\n"
    nmlstr += f"    kstrength = 0\n"
    nmlstr += f"    krdg_partic = 1\n"
    nmlstr += f"    krdg_redist = 1\n"
    nmlstr += f"    mu_rdg = 3\n"
    nmlstr += f"    cf = 17.0\n"
    nmlstr += f"/\n"
    nmlstr += f"\n"
    nmlstr += f"&shortwave_nml\n"
    nmlstr += f"    shortwave = 'dEdd'\n"
    nmlstr += f"    albedo_type = 'constant'\n"
    nmlstr += f"    albicev = 0.78\n"
    nmlstr += f"    albicei = 0.36\n"
    nmlstr += f"    albsnowv = 0.98\n"
    nmlstr += f"    albsnowi = 0.7\n"
    nmlstr += f"    ahmax = 0.3\n"
    nmlstr += f"    r_ice = 1.5\n"
    nmlstr += f"    r_pnd = 1.2\n"
    nmlstr += f"    r_snw = 1.2\n"
    nmlstr += f"    dt_mlt = 1.5\n"
    nmlstr += f"    rsnw_mlt = 1500.0\n"
    nmlstr += f"    kalg = 0.6\n"
    nmlstr += f"/\n"
    nmlstr += f"\n"
    nmlstr += f"&ponds_nml\n"
    nmlstr += f"    hp1 = 0.01\n"
    nmlstr += f"    hs0 = 0.0\n"
    nmlstr += f"    hs1 = 0.03\n"
    nmlstr += f"    dpscale = 0.001\n"
    nmlstr += f"    frzpnd = 'hlid'\n"
    nmlstr += f"    rfracmin = 0.15\n"
    nmlstr += f"    rfracmax = 1.0\n"
    nmlstr += f"    pndaspect = 0.8\n"
    nmlstr += f"/\n"
    nmlstr += f"\n"
    nmlstr += f"&zbgc_nml\n"
    nmlstr += f"    tr_brine = .false.\n"
    nmlstr += f"    restart_hbrine = .false.\n"
    nmlstr += f"    skl_bgc = .false.\n"
    nmlstr += f"    bgc_flux_type = 'Jin2006'\n"
    nmlstr += f"    restart_bgc = .false.\n"
    nmlstr += f"    restore_bgc = .false.\n"
    nmlstr += f"    bgc_data_dir = 'unknown_bgc_data_dir'\n"
    nmlstr += f"    sil_data_type = 'default'\n"
    nmlstr += f"    nit_data_type = 'default'\n"
    nmlstr += f"    tr_bgc_c_sk = .false.\n"
    nmlstr += f"    tr_bgc_chl_sk = .false.\n"
    nmlstr += f"    tr_bgc_am_sk = .false.\n"
    nmlstr += f"    tr_bgc_sil_sk = .false.\n"
    nmlstr += f"    tr_bgc_dmspp_sk = .false.\n"
    nmlstr += f"    tr_bgc_dmspd_sk = .false.\n"
    nmlstr += f"    tr_bgc_dms_sk = .false.\n"
    nmlstr += f"    phi_snow = 0.5\n"
    nmlstr += f"/\n"
    nmlstr += f"\n"
    nmlstr += f"&forcing_nml\n"
    nmlstr += f"    formdrag = .false.\n"
    nmlstr += f"    atmbndy = 'default'\n"
    nmlstr += f"    fyear_init = {year_init}\n"
    nmlstr += f"    ycycle = 52\n"
    nmlstr += f"    atm_data_format = 'bin'\n"
    nmlstr += f"    atm_data_type = 'None'\n"
    nmlstr += f"    atm_data_dir = '/usr/projects/climate/eclare/DATA/gx1v3/LargeYeager/v2_updated/'\n"
    nmlstr += f"    calc_strair = .true.\n"
    nmlstr += f"    highfreq = .false.\n"
    nmlstr += f"    natmiter = 5\n"
    nmlstr += f"    calc_tsfc = .true.\n"
    nmlstr += f"    precip_units = 'mm_per_sec'\n"
    nmlstr += f"    ustar_min = 0.0005\n"
    nmlstr += f"    fbot_xfer_type = 'constant'\n"
    nmlstr += f"    update_ocn_f = .true.\n"
    nmlstr += f"    l_mpond_fresh = .false.\n"
    nmlstr += f"    tfrz_option = 'linear_salt'\n"
    nmlstr += f"    oceanmixed_ice = .false.\n"
    nmlstr += f"    ocn_data_format = 'nc'\n"
    nmlstr += f"    sss_data_type = 'default'\n"
    nmlstr += f"    sst_data_type = 'default'\n"
    nmlstr += f"    ocn_data_dir = '/usr/projects/climate/eclare/DATA/gx1v3/gx1v3/forcing/'\n"
    nmlstr += f"    oceanmixed_file = 'oceanmixed_ice_depth.nc'\n"
    nmlstr += f"    restore_sst = .false.\n"
    nmlstr += f"    trestore = 90\n"
    nmlstr += f"    restore_ice = .false.\n"
    nmlstr += f"/\n"
    nmlstr += f"\n"
    nmlstr += f"&icefields_nml\n"
    nmlstr += f"    f_tmask = .true.\n"
    nmlstr += f"    f_blkmask = .true.\n"
    nmlstr += f"    f_tarea = .true.\n"
    nmlstr += f"    f_uarea = .true.\n"
    nmlstr += f"    f_dxt = .false.\n"
    nmlstr += f"    f_dyt = .false.\n"
    nmlstr += f"    f_dxu = .false.\n"
    nmlstr += f"    f_dyu = .false.\n"
    nmlstr += f"    f_htn = .false.\n"
    nmlstr += f"    f_hte = .false.\n"
    nmlstr += f"    f_angle = .true.\n"
    nmlstr += f"    f_anglet = .true.\n"
    nmlstr += f"    f_ncat = .true.\n"
    nmlstr += f"    f_vgrdi = .false.\n"
    nmlstr += f"    f_vgrds = .false.\n"
    nmlstr += f"    f_vgrdb = .false.\n"
    nmlstr += f"    f_bounds = .false.\n"
    nmlstr += f"    f_aice = 'd'\n"
    nmlstr += f"    f_hi = 'd'\n"
    nmlstr += f"    f_hs = 'd'\n"
    nmlstr += f"    f_tsfc = 'd'\n"
    nmlstr += f"    f_sice = 'd'\n"
    nmlstr += f"    f_uvel = 'd'\n"
    nmlstr += f"    f_vvel = 'd'\n"
    nmlstr += f"    f_uatm = 'h'\n"
    nmlstr += f"    f_vatm = 'h'\n"
    nmlstr += f"    f_fswdn = 'h'\n"
    nmlstr += f"    f_flwdn = 'h'\n"
    nmlstr += f"    f_snow = 'd'\n"
    nmlstr += f"    f_snow_ai = 'm'\n"
    nmlstr += f"    f_rain = 'h'\n"
    nmlstr += f"    f_rain_ai = 'm'\n"
    nmlstr += f"    f_sst = 'd'\n"
    nmlstr += f"    f_sss = 'd'\n"
    nmlstr += f"    f_uocn = 'd'\n"
    nmlstr += f"    f_vocn = 'd'\n"
    nmlstr += f"    f_frzmlt = 'h'\n"
    nmlstr += f"    f_fswfac = 'm'\n"
    nmlstr += f"    f_fswint_ai = 'm'\n"
    nmlstr += f"    f_fswabs = 'x'\n"
    nmlstr += f"    f_fswabs_ai = 'm'\n"
    nmlstr += f"    f_albsni = 'm'\n"
    nmlstr += f"    f_alvdr = 'x'\n"
    nmlstr += f"    f_alidr = 'x'\n"
    nmlstr += f"    f_alvdf = 'x'\n"
    nmlstr += f"    f_alidf = 'x'\n"
    nmlstr += f"    f_albice = 'd'\n"
    nmlstr += f"    f_albsno = 'd'\n"
    nmlstr += f"    f_albpnd = 'd'\n"
    nmlstr += f"    f_coszen = 'x'\n"
    nmlstr += f"    f_flat = 'x'\n"
    nmlstr += f"    f_flat_ai = 'm'\n"
    nmlstr += f"    f_fsens = 'x'\n"
    nmlstr += f"    f_fsens_ai = 'm'\n"
    nmlstr += f"    f_flwup = 'h'\n"
    nmlstr += f"    f_flwup_ai = 'h'\n"
    nmlstr += f"    f_evap = 'x'\n"
    nmlstr += f"    f_evap_ai = 'm'\n"
    nmlstr += f"    f_tair = 'h'\n"
    nmlstr += f"    f_tref = 'x'\n"
    nmlstr += f"    f_qref = 'x'\n"
    nmlstr += f"    f_congel = 'm'\n"
    nmlstr += f"    f_frazil = 'm'\n"
    nmlstr += f"    f_snoice = 'm'\n"
    nmlstr += f"    f_dsnow = 'x'\n"
    nmlstr += f"    f_melts = 'm'\n"
    nmlstr += f"    f_meltt = 'm'\n"
    nmlstr += f"    f_meltb = 'm'\n"
    nmlstr += f"    f_meltl = 'm'\n"
    nmlstr += f"    f_fresh = 'h'\n"
    nmlstr += f"    f_fresh_ai = 'h'\n"
    nmlstr += f"    f_fsalt = 'h'\n"
    nmlstr += f"    f_fsalt_ai = 'h'\n"
    nmlstr += f"    f_fhocn = 'h'\n"
    nmlstr += f"    f_fhocn_ai = 'h'\n"
    nmlstr += f"    f_fswthru = 'h'\n"
    nmlstr += f"    f_fswthru_ai = 'h'\n"
    nmlstr += f"    f_fsurf_ai = 'h'\n"
    nmlstr += f"    f_fcondtop_ai = 'x'\n"
    nmlstr += f"    f_fmeltt_ai = 'h'\n"
    nmlstr += f"    f_strairx = 'h'\n"
    nmlstr += f"    f_strairy = 'h'\n"
    nmlstr += f"    f_strtltx = 'h'\n"
    nmlstr += f"    f_strtlty = 'h'\n"
    nmlstr += f"    f_strcorx = 'x'\n"
    nmlstr += f"    f_strcory = 'x'\n"
    nmlstr += f"    f_strocnx = 'h'\n"
    nmlstr += f"    f_strocny = 'h'\n"
    nmlstr += f"    f_strintx = 'x'\n"
    nmlstr += f"    f_strinty = 'x'\n"
    nmlstr += f"    f_strength = 'm'\n"
    nmlstr += f"    f_divu = 'm'\n"
    nmlstr += f"    f_shear = 'm'\n"
    nmlstr += f"    f_sig1 = 'm'\n"
    nmlstr += f"    f_sig2 = 'm'\n"
    nmlstr += f"    f_dvidtt = 'm'\n"
    nmlstr += f"    f_dvidtd = 'm'\n"
    nmlstr += f"    f_daidtt = 'm'\n"
    nmlstr += f"    f_daidtd = 'm'\n"
    nmlstr += f"    f_dagedtt = 'm'\n"
    nmlstr += f"    f_dagedtd = 'm'\n"
    nmlstr += f"    f_mlt_onset = 'm'\n"
    nmlstr += f"    f_frz_onset = 'm'\n"
    nmlstr += f"    f_hisnap = 'x'\n"
    nmlstr += f"    f_aisnap = 'x'\n"
    nmlstr += f"    f_trsig = 'm'\n"
    nmlstr += f"    f_icepresent = 'h'\n"
    nmlstr += f"    f_iage = 'm'\n"
    nmlstr += f"    f_fy = 'm'\n"
    nmlstr += f"    f_aicen = 'x'\n"
    nmlstr += f"    f_vicen = 'x'\n"
    nmlstr += f"    f_vsnon = 'x'\n"
    nmlstr += f"    f_keffn_top = 'x'\n"
    nmlstr += f"    f_tinz = 'x'\n"
    nmlstr += f"    f_sinz = 'x'\n"
    nmlstr += f"    f_tsnz = 'x'\n"
    nmlstr += f"    f_fsurfn_ai = 'h'\n"
    nmlstr += f"    f_fcondtopn_ai = 'h'\n"
    nmlstr += f"    f_fmelttn_ai = 'h'\n"
    nmlstr += f"    f_flatn_ai = 'x'\n"
    nmlstr += f"    f_fsensn_ai = 'x'\n"
    nmlstr += f"/\n"
    nmlstr += f"\n"
    nmlstr += f"&icefields_mechred_nml\n"
    nmlstr += f"    f_alvl = 'm'\n"
    nmlstr += f"    f_vlvl = 'm'\n"
    nmlstr += f"    f_ardg = 'm'\n"
    nmlstr += f"    f_vrdg = 'm'\n"
    nmlstr += f"    f_dardg1dt = 'x'\n"
    nmlstr += f"    f_dardg2dt = 'x'\n"
    nmlstr += f"    f_dvirdgdt = 'x'\n"
    nmlstr += f"    f_opening = 'x'\n"
    nmlstr += f"    f_ardgn = 'x'\n"
    nmlstr += f"    f_vrdgn = 'x'\n"
    nmlstr += f"    f_dardg1ndt = 'x'\n"
    nmlstr += f"    f_dardg2ndt = 'x'\n"
    nmlstr += f"    f_dvirdgndt = 'x'\n"
    nmlstr += f"    f_krdgn = 'x'\n"
    nmlstr += f"    f_aparticn = 'x'\n"
    nmlstr += f"    f_aredistn = 'x'\n"
    nmlstr += f"    f_vredistn = 'x'\n"
    nmlstr += f"    f_araftn = 'x'\n"
    nmlstr += f"    f_vraftn = 'x'\n"
    nmlstr += f"/\n"
    nmlstr += f"\n"
    nmlstr += f"&icefields_pond_nml\n"
    nmlstr += f"    f_apondn = 'x'\n"
    nmlstr += f"    f_apeffn = 'x'\n"
    nmlstr += f"    f_hpondn = 'x'\n"
    nmlstr += f"    f_apond = 'm'\n"
    nmlstr += f"    f_hpond = 'm'\n"
    nmlstr += f"    f_ipond = 'm'\n"
    nmlstr += f"    f_apeff = 'm'\n"
    nmlstr += f"    f_apond_ai = 'm'\n"
    nmlstr += f"    f_hpond_ai = 'm'\n"
    nmlstr += f"    f_ipond_ai = 'm'\n"
    nmlstr += f"    f_apeff_ai = 'm'\n"
    nmlstr += f"/\n"
    nmlstr += f"\n"
    nmlstr += f"&icefields_bgc_nml\n"
    nmlstr += f"    f_faero_atm = 'x'\n"
    nmlstr += f"    f_faero_ocn = 'x'\n"
    nmlstr += f"    f_aero = 'x'\n"
    nmlstr += f"    f_fno = 'x'\n"
    nmlstr += f"    f_fno_ai = 'x'\n"
    nmlstr += f"    f_fnh = 'x'\n"
    nmlstr += f"    f_fnh_ai = 'x'\n"
    nmlstr += f"    f_fn = 'x'\n"
    nmlstr += f"    f_fn_ai = 'x'\n"
    nmlstr += f"    f_fsil = 'x'\n"
    nmlstr += f"    f_fsil_ai = 'x'\n"
    nmlstr += f"    f_bgc_n_sk = 'x'\n"
    nmlstr += f"    f_bgc_c_sk = 'x'\n"
    nmlstr += f"    f_bgc_chl_sk = 'x'\n"
    nmlstr += f"    f_bgc_nit_sk = 'x'\n"
    nmlstr += f"    f_bgc_am_sk = 'x'\n"
    nmlstr += f"    f_bgc_sil_sk = 'x'\n"
    nmlstr += f"    f_bgc_dmspp_sk = 'x'\n"
    nmlstr += f"    f_bgc_dmspd_sk = 'x'\n"
    nmlstr += f"    f_bgc_dms_sk = 'x'\n"
    nmlstr += f"    f_bgc_nit_ml = 'x'\n"
    nmlstr += f"    f_bgc_am_ml = 'x'\n"
    nmlstr += f"    f_bgc_sil_ml = 'x'\n"
    nmlstr += f"    f_bgc_dmsp_ml = 'x'\n"
    nmlstr += f"    f_btin = 'x'\n"
    nmlstr += f"    f_bphi = 'x'\n"
    nmlstr += f"    f_fbri = 'm'\n"
    nmlstr += f"    f_hbri = 'm'\n"
    nmlstr += f"    f_grownet = 'x'\n"
    nmlstr += f"    f_ppnet = 'x'\n"
    nmlstr += f"/\n"
    nmlstr += f"\n"
    nmlstr += f"&icefields_drag_nml\n"
    nmlstr += f"    f_drag = 'x'\n"
    nmlstr += f"    f_cdn_atm = 'x'\n"
    nmlstr += f"    f_cdn_ocn = 'x'\n"
    nmlstr += f"/\n"
    return nmlstr

def namelist(m, time, forecast_period, run_dir='.'):
    """
    Generate namelists for TP5 model runs,
    Inputs:
    - m: Model object with configs
    - run_dir: path to the run directory
    """
    with open(os.path.join(run_dir, 'blkdat.input'), 'wt') as f:
        f.write(blkdat(m))

    with open(os.path.join(run_dir, 'limits'), 'wt') as f:
        f.write(limits(m, time, forecast_period))

    with open(os.path.join(run_dir, 'ports.input'), 'wt') as f:
        f.write(ports(m))

    with open(os.path.join(run_dir, 'ice_in'), 'wt') as f:
        f.write(ice_in(m, time, forecast_period))

