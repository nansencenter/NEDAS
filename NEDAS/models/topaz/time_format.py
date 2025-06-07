from datetime import datetime, timedelta, timezone

def diy(iyr):
   if ( (iyr%4 == 0 and 1901+iyr%400==0) or (iyr % 4 != 0 )):
      days=365
   else:
      days=366
   return days

def forday(yrflag,dtime):
   ## converts model day to "calendar" date (year,ordinal-day,hour).
   if (yrflag==0):
      #  360 days per model year, starting Jan 16
      iyear =  int((dtime+15.001)/360) + 1
      iday  =  (dtime+15.001) % 360. + 1
      ihour = ( ( dtime+15.001) % 360. + 1. - iday)*24.
   elif (yrflag==1):
      # 366 days per model year, starting Jan 16
      iyear =  int((dtime+15.001)/366.) + 1
      iday  =  ( dtime+15.001 ) % 366. + 1
      ihour =  ( ( dtime+15.001 ) % 366. + 1. - iday)*24.
   elif (yrflag==2):
      # 366 days per model year, starting Jan 01
      iyear =  int((dtime+ 0.001)/366.) + 1.
      iday  =  ( dtime+ 0.001 ) % 366. + 1.
      ihour = ( ( dtime+ 0.001 ) % 366. + 1. - iday)*24.
   elif (yrflag==3):
      # model day is calendar days since 01/01/1901
      iyr   = int((dtime-1.)/365.25)
      nleap = iyr/4
      dtim1 = 365.*iyr + nleap + 1.
      day   = dtime - dtim1 + 1.
      if     (dtim1>dtime):
         iyr = iyr - 1
      elif (day>=367.):
         iyr = iyr + 1
      elif ( (day>=366.) and (iyr % 4 != 3) ):
         iyr = iyr + 1
      nleap = int(iyr/4)
      dtim1 = 365.*iyr + nleap + 1.
      iyear =  1901 + iyr
      iday  =  dtime - dtim1 + 1.001
      ihour = (dtime - dtim1 + 1.001 - iday)*24
   else:
      print("unknown flag")
   return  int(iyear) , int(iday), int(ihour)

def dayfor(yrflag, iyear,iday,ihour):
   ##converts "calendar" date (year,ordinal-day,hour) to model day
   if     (yrflag==0):
      # 360 days per model year, starting Jan 16
      dtime =  (iyear-1) * 360. +  iday  + ihour/24. - 16.
   elif (yrflag==1):
      # 366 days per model year, starting Jan 16
        dtime = (iyear-1) * 366. + iday  + ihour/24.- 16.
   elif (yrflag==2):
      # 366 days per model year, starting Jan 01
      dtime = (iyear-1) * 366. + iday   - ihour/24. - 1.
   elif (yrflag==3):
      #model day is calendar days since 01/01/1901
      iyr=iyear-1901
      dtime=0.
      for k in range (iyr):
         dtime=dtime+diy(k+1901)

      iday2=iday
      while (iday2+ihour/24.>0.):
         iyr=iyr+1
         diy2=diy(iyr+1901)

         if ((iday2+(ihour/24.))/diy2>=1.):
            dtime=dtime+diy2
         else:
            dtime=dtime+iday2+ihour/24.
         iday2=iday2-diy2
   return dtime

def datetojul(t, refyear=1950, refmonth=1, refday=1):
    ref_t = datetime(refyear, refmonth, refday, tzinfo=timezone.utc)
    jday = (t - ref_t) / timedelta(days=1)
    return jday

def jultodate(jday, refyear=1950, refmonth=1, refday=1):
    ref_t = datetime(refyear, refmonth, refday, tzinfo=timezone.utc)
    t = ref_t + jday * timedelta(days=1)
    return t

