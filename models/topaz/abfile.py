###adapted from NERSC-HYCOM-CICE/pythonlibs/abfile
##Module for doing IO on files used by hycom
import numpy
import struct
import sys
import re

# Firldnames as they appear in ordered form in the regional grid files
grid_ordered_fieldnames = [
    "plon", "plat", "qlon", "qlat", "ulon", "ulat", "vlon", "vlat", "pang", "scpx",
    "scpy", "scqx", "scqy", "scux", "scuy", "scvx", "scvy", "cori", "pasp"
    ]

class AFileError(Exception) :
    pass

class BFileError(Exception) :
    pass

class AFile(object) :
    """ Class for doing binary input/output on hycom .a files. Normally used by 
    ABFile* classes, but can be called by itself to read a .a-file """
    huge = 2.0**100
    def __init__(self,idm,jdm,filename,action,mask=False,real4=True,endian="big") :
        self._idm = idm
        self._jdm = jdm
        self._filename = filename
        self._action = action
        self._mask = mask
        self._real4= real4
        self._endian= endian
        if self._action.lower() not in ["r","w","r+"]  :
            raise AFileError("action argument must be either r(ead) or w(rite) or r+")
        if self._endian.lower() not in ["little","big","native"]  :
            raise AFileError("action argument must be either native, little or big")

        if self._endian.lower() == "native" :
            self._endian=sys.byteorder

        if self._endian.lower() == "big" :
            self._endian_structfmt = ">"
        else :
            self._endian_structfmt = "<"

        self._init_record()
        self._open()

    def _init_record(self) :
        # Size of output 2D Array
        self._n2drec= ((self._idm*self._jdm+4095)//4096)*4096 # original

        # Init sequenctial record counter
        self._iarec = 0
        self._spval = 2**100.

    def _open(self) :
        # Open .a and .b file
        self._filea = open(self._filename,self._action+"b")

    def writerecord(self,h,mask=None,record=None) :
        """ Write one record to file.

        Arguments:    
          h      : data field to write. Must conform to shape (self._jdm, self._idm)
          mask  : where mask is set to True, h is set to spval. Ignored if masking not set in class

        Keyword Arguments:
          record:  not implemented yet
        """

        # Initialize writing array (1D)
        w=numpy.ones(self._n2drec) *self._spval

        # Check array shape against idm/jdm
        if h.shape[0] != self._jdm or h.shape[1] != self._idm :
            raise AFileError("array shape is (%d,%d),expected (%d,%d)"%(h.shape[0],h.shape[1],self._jdm,self._idm))

        # Fill w array
        w[0:self._idm*self._jdm] = h.flatten()

        # Seek if provided, otherwise use current position
        if record is not None : self.seekrecord(record)

        # Calc min and mask
        #print("mask boolean:",self._mask)
        if self._mask :
            if mask is None:
                mask = np.isnan(h)
            I=numpy.where(~mask)
            hmax=h[I].max()
            hmin=h[I].min()
            J=numpy.where(mask.flatten())
            w[J] = self._spval
          # print "writerecord w mask:",numpy.count_nonzero(mask),mask.size
          # print "writerecord w mask:",hmin,hmax,w[0:self._idm*self._jdm].min(),w[0:self._idm*self._jdm].max()
        else :
            hmax=h.max()
            hmin=h.min()
            #print "writerecord wo mask:",hmin,hmax

        if self._real4 :
            struct_fmt="f"
        else :
            struct_fmt="d"

        # 1) Use struct
        #binpack=struct.pack("%s%d%s"%(self._endian_structfmt,w.size,struct_fmt),*w[:])
        #self._filea.write(binpack)

        # 2) Alternative using numpy.ndarray.tofile
        w = w.astype(numpy.dtype("%s%s"%(self._endian_structfmt,struct_fmt)))
        #print w.dtype,w.dtype.byteorder
        w.tofile(self._filea)




        return hmin,hmax

    def read_record(self,record) :
        self.seekrecord(record)

        ##1) Use struct
        #if self._real4 :
        #    raw = self._filea.read(self.n2drec*4)
        #    fmt =  "%s%df"%(self._endian_structfmt,self.n2drec)
        #else :
        #    raw = self._filea.read(self.n2drec*8)
        #    fmt =  "%s%dd"%(self._endian_structfmt,self.n2drec)
        #w =  numpy.array(struct.unpack(fmt,raw))

        #2) Use numpy fromfile
        if self._real4 :
            struct_fmt="f"
        else :
            struct_fmt="d"
        mydtype=numpy.dtype("%s%s"%(self._endian_structfmt,struct_fmt))
        w=numpy.fromfile(self._filea,dtype=mydtype,count=int(self.n2drec))

        w=w[0:self.idm*self.jdm]
        w.shape=(self.jdm,self.idm)
        #print w.min(),w.max()
        w=numpy.ma.masked_where(w>self.huge*.5,w)
        #print w.min(),w.max()


        return w

    def seekrecord(self,record) :
        # Seek to correct record and read
        if self._real4 :
            self._filea.seek(int(record*self.n2drec*4))
        else :
            self._filea.seek(int(record*self.n2drec*8))
        return

    def close(self) :
        self._filea.close()

    @property
    def n2drec(self):
        return self._n2drec

    @property
    def idm(self):
        return self._idm

    @property
    def jdm(self):
        return self._jdm

class ABFile(object) :
    """ Class for doing input/output on pairs of hycom .a and .b files. Base class, 
    not meant to be used directly """

    def __init__(self,basename,action,mask=False,real4=True,endian="big") :
        #self._basename=basename
        self._basename=ABFile.strip_ab_ending(basename) # Ensure .ab ending is stripped
        #print basename,self._basename
        self._action=action
        self._fileb = open(self._basename+".b",self._action)
        self._filea = None
        self._mask = mask
        self._real4 = real4
        self._endian = endian
        self._firstwrite=True

    def close(self) :
        self._fileb.close()

    def scanitem(self,item=None,conversion=None) :
        line = self._fileb.readline().strip()
        if item is not None :
            pattern="^(.*)'(%-6s)'[ =]*"%item
            m=re.match(pattern,line)
        else :
            m=re.match("^(.*)'(.*)'[ =]*",line)
        if m :
            if conversion :
                value = conversion(m.group(1))
            return m.group(2),value
        else :
            return None,None

    def writeitem(self,key,value) :
        if value is None :
            msg = "key %s is not defined "%key
            raise ValueError(msg)
        if type(value) == type(1) :
            tmp ="%5d    '%-6s'\n"%(value,key)
        else :
            msg = "writeitem not implemented for this type: %s"%type(value)
            raise NotImplementedError(msg)
        self._fileb.write(tmp)

    def readline(self) :
        return self._fileb.readline()

    def bminmax(self,*arks,**kwargs) :
        raise BFileError("bminmax not implemented for this class")

    @property
    def fieldnames(self) :
        return set([elem["field"] for elem in self._fields.values()])

    def write_field(*args,**kwargs) :
        raise BFileError("write_field not implemented for this class")

    def read_field(*args,**kwargs) :
        raise BFileError("read_field not implemented for this class")

    def write_header(self,*args,**kwargs) :
        raise BFileError("write_header not implemented for this class")

    def read_header(self,*args,**kwargs) :
        raise BFileError("read_header not implemented for this class")

    def _open_filea_if_necessary(self,field) :
        if self._filea is None :
            self._jdm,self._idm = field.shape
            self._filea = AFile(self._idm,self._jdm,self._basename+".a",
                    self._action,mask=self._mask,real4=self._real4,endian=self._endian)
        else :
            pass

    def close (self):
        self._filea.close()
        self._fileb.close()

    def check_dimensions(self,field) :
        if self._idm != field.shape[1] or self._jdm != field.shape[0] :
            msg = "dimensions do not match for field. Field dim=%dX%d, file dim=%dX%d"%(field.shape[0].field.shape[1], self._idm, self._jdm)
            raise BFileError(msg)

    @property
    def idm(self):
        return self._idm

    @property
    def jdm(self):
        return self._jdm

    @property
    def fields(self) :
        return self._fields

    @classmethod
    def strip_ab_ending(cls,fname) :
        m=re.match( "^(.*)(\.[ab]$)", fname)
        if m :
            return m.group(1)
        else :
            return fname

    @classmethod
    def check_minmax(cls,w,mydict) :
#      if (abs(hmina(k)-hminb).gt.abs(hminb)*1.e-4 .or.
#                                &             abs(hmaxa(k)-hmaxb).gt.abs(hmaxb)*1.e-4      ) then
        testmin = numpy.abs(w.min()-mydict["min"]) > numpy.abs(mydict["min"])*1.e-4 
        testmax = numpy.abs(w.max()-mydict["max"]) > numpy.abs(mydict["max"])*1.e-4 
        if testmin or testmax :
            msg="File a b values inconsistent: Field %s at %d, .a min/max: %12.6g %12.6g  .a min/max: %12.6g  %12.6g " %(
                mydict["field"],mydict["k"],w.min(),w.max(),mydict["min"],mydict["max"])
            raise ValueError(msg)

    @property
    def basename(self) :
        return self._basename

class ABFileBathy(ABFile) :
    """ Class for doing input/output on pairs of hycom .a and .b files. This is for bathymetry files"""
    def __init__(self,basename,action,mask=True,real4=True,endian="big",idm=None,jdm=None) :

        super(ABFileBathy,self).__init__(basename,action,mask=mask,real4=real4,endian=endian)
        if action == "r" :
            if idm != None and jdm != None:
                self._idm=idm
                self._jdm=jdm
                self.read_header()
                self.read_field_info()
                self._open_filea_if_necessary(numpy.zeros((self._jdm,self._idm)))
            else :
                raise BFileError("ABFileBathy opened as read, but idm and jdm not provided")
        else :
            self.write_header()

    def write_header(self) :
        self._fileb.write("Bathymetry prepared by python modeltools package\n")
        self._fileb.write("\n")
        self._fileb.write("\n")
        self._fileb.write("\n")
        self._fileb.write("\n")

    def read_header(self) :
        self._header=[]
        self._header.append(self.readline())
        self._header.append(self.readline())
        self._header.append(self.readline())
        self._header.append(self.readline())
        self._header.append(self.readline())

    def read_field_info(self) :
        fieldkeys=["field","min","max"]
        self._fields={}
        line=self.readline().strip()
        i=0
        while line :
            m = re.match("^min,max[ ]+(.*)[ ]*=(.*)",line)
            if m :
                self._fields[i] = {}
                self._fields[i]["field"] = m.group(1).strip()
                elem = [elem.strip() for elem in m.group(2).split() if elem.strip()]
                self._fields[i]["min"] = float(elem[0])
                self._fields[i]["max"] = float(elem[1])
            i+=1
            line=self.readline().strip()

    def write_field(self,field,mask) : 
        self._open_filea_if_necessary(field)
        hmin,hmax = self._filea.writerecord(field,mask,record=None)
        self._fileb.write("min,max %s =%16.5f%16.5f\n"%("depth",hmin,hmax))

    def read_field(self,fieldname) :
        """ Read field corresponding to fieldname and level from bathy file"""
        #print self._fields
        record = None
        for i,d in self._fields.items() :
            if d["field"] == fieldname :
                record=i
        if record  is not None :
            w = self._filea.read_record(record) 
        else :
            w = None
        return w

    def bminmax(self,fieldname) :
        record=None
        for i,d in self._fields.items() :
            if d["field"] == fieldname :
                record=i
        if record  is not None :
            ret = (self._fields[i]["min"],self._fields[i]["max"])
        else :
            ret = (None,None)
        return ret

class ABFileRmu(ABFile) :
    """ Class for doing input/output on pairs of hycom .a and .b files. This is for nesting/relax rmu files"""
    def __init__(self,basename,action,mask=False,real4=True,endian="big",idm=None,jdm=None,cline1="",cline2="") :

        super(ABFileRmu,self).__init__(basename,action,mask=mask,real4=real4,endian=endian)
        self._cline1=cline1
        self._cline2=cline2
        if action == "w" :
            pass
        else :
            self.read_header()
            self.read_field_info()
            self._open_filea_if_necessary(numpy.zeros((self._jdm,self._idm)))

    def write_header(self) :
        self._fileb.write("%s\n"%self._cline1)
        self._fileb.write("%s\n"%self._cline2)
        self._fileb.write("\n")
        self._fileb.write("\n")
        self._fileb.write("i/jdm =%5d %5d\n"%(self._idm,self._jdm))

    def read_header(self) :
        self._header=[]
        self._header.append(self.readline())
        self._header.append(self.readline())
        self._header.append(self.readline())
        self._header.append(self.readline())
        self._header.append(self.readline())
        m = re.match("i/jdm[ ]*=[ ]*([0-9]+)[ ]+([0-9]+)",self._header[4].strip())
        if m :
            self._idm = int(m.group(1))
            self._jdm = int(m.group(2))
        else :
            raise  AFileError("Unable to parse idm, jdm from header. File=%s, Parseable string=%s"%(
                    self._filename, self._header[4].strip()))

    def read_field_info(self) :
        fieldkeys=["field","min","max"]
        self._fields={}
        line=self.readline().strip()
        i=0
        while line :
            m = re.match("^min,max[ ]+(.*)[ ]*=(.*)",line)
            if m :
                self._fields[i] = {}
                self._fields[i]["field"] = m.group(1).strip()
                elem = [elem.strip() for elem in m.group(2).split() if elem.strip()]
                self._fields[i]["min"] = float(elem[0])
                self._fields[i]["max"] = float(elem[1])
            i+=1
            line=self.readline().strip()

    def write_field(self,field,mask,fieldname,fmt="%16.8g") :
        self._open_filea_if_necessary(field)
        if self._firstwrite :
            self._jdm,self._idm=field.shape
            self._firstwrite=False
            self.write_header()
        self.check_dimensions(field)
        hmin,hmax = self._filea.writerecord(field,mask)
        fmtstr="%%4s:  min,max =%s %s\n"%(fmt,fmt)
        self._fileb.write(fmtstr%(fieldname,hmin,hmax))

    def read_field(self,fieldname) :
        """ Read field corresponding to fieldname and level from bathy file"""
        #print self._fields
        record = None
        for i,d in self._fields.items() :
            if d["field"] == fieldname :
                record=i
        if record  is not None :
            w = self._filea.read_record(record)
        else :
            w = None
        return w

    def bminmax(self,fieldname) :
        record=None
        for i,d in self._fields.items() :
            if d["field"] == fieldname :
                record=i
        if record  is not None :
            ret = (self._fields[i]["min"],self._fields[i]["max"])
        else :
            ret = (None,None)
        return ret

class ABFileGrid(ABFile) :
    """ Class for doing input/output on pairs of hycom .a and .b files. This is for grid files"""
    fieldkeys=["min","max"]
    def __init__(self,basename,action,mask=False,real4=True,endian="big",mapflg=-1) :

        super(ABFileGrid,self).__init__(basename,action,mask=mask,real4=real4,endian=endian)
        self._mapflg=mapflg

        if action == "w" :
            pass
        else :
            self.read_header()
            self.read_field_info()
            self._open_filea_if_necessary(numpy.zeros((self._jdm,self._idm)))

    def read_header(self) :
        item,self._idm     = self.scanitem(item="idm",conversion=int)
        item,self._jdm     = self.scanitem(item="jdm",conversion=int)
        item,self._mapflg = self.scanitem(item="mapflg",conversion=int)

    def write_header(self) :
        self.writeitem("idm",self._idm)
        self.writeitem("jdm",self._jdm)
        self.writeitem("mapflg",self._mapflg)

    def read_field_info(self) :
        # Get list of fields from .b file
        #plon:  min,max =        -179.99806         179.99998
        #plat:  min,max =         -15.79576          89.98227
        #...
        self._fields={}
        line=self.readline().strip()
        i=0
        while line :
            fieldname = line[0:4]

            self._fields[i]={}
            self._fields[i]["field"]=fieldname
            elems = re.split("[ =]+",line)
            self._fields[i]["min"]=elems[2]
            self._fields[i]["max"]=elems[3]
            for k in self.fieldkeys :
                self._fields[i][k] = float(self._fields[i][k])
            i+=1
            line=self.readline().strip()

    def read_field(self,fieldname) :
        """ Read field corresponding to fieldname and level from archive file"""
        record = None
        for i,d in self._fields.items() :
            if d["field"] == fieldname :
                record=i
        if record  is not None :
            w = self._filea.read_record(record) 
        else :
            w = None
        return w

    def write_field(self,field,mask,fieldname,fmt="%16.8g") :
        self._open_filea_if_necessary(field)
        if self._firstwrite :
            self._jdm,self._idm=field.shape
            self._firstwrite=False
            self.write_header()
        self.check_dimensions(field)
        hmin,hmax = self._filea.writerecord(field,mask)
        fmtstr="%%4s:  min,max =%s %s\n"%(fmt,fmt)
        self._fileb.write(fmtstr%(fieldname,hmin,hmax))

    def bminmax(self,fieldname) :
        record=None
        for i,d in self._fields.items() :
            if d["field"] == fieldname :
                record=i
        if record  is not None :
            ret = (self._fields[i]["min"],self._fields[i]["max"])
        else :
            ret = (None,None)
        return ret

class ABFileArchv(ABFile) :
    """ Class for doing input/output on pairs of hycom .a and .b files. This is for archv files"""
    fieldkeys=["field","step","day","k","dens","min","max"]
    def __init__(self,basename,action,mask=True,real4=True,endian="big",
            iversn=None,iexpt=None,yrflag=None,cline1="",cline2="",cline3="") :

        self._cline1=cline1
        self._cline2=cline2
        self._cline3=cline3
        self._iversn=iversn
        self._iexpt =iexpt 
        self._yrflag=yrflag

        super(ABFileArchv,self).__init__(basename,action,mask=mask,real4=real4,endian=endian)
        if self._action == "r" :
            self.read_header() # Sets internal metadata. Overrides those on input
            self.read_field_info()
            self._open_filea_if_necessary(numpy.zeros((self._jdm,self._idm)))
        elif self._action == "w" :
            ## Need to test if idm, jdm, etc is set at this stage
            #raise NotImplementedError,"ABFileArchv writing not implemented"
            pass

    def read_header(self) :
        self._header=[]
        self._header.append(self.readline())
        self._header.append(self.readline())
        self._header.append(self.readline())
        self._header.append(self.readline())

        item,self._iversn = self.scanitem(item="iversn",conversion=int)
        item,self._iexpt  = self.scanitem(item="iexpt",conversion=int)
        item,self._yrflag = self.scanitem(item="yrflag",conversion=int)
        item,self._idm     = self.scanitem(item="idm",conversion=int)
        item,self._jdm     = self.scanitem(item="jdm",conversion=int)

    def read_field_info(self) :
        # Get list of fields from .b file
        #field         time step  model day  k  dens          min                  max
        #montg1    =        67392     351.000  1 25.000    0.0000000E+00    0.0000000E+00
        #
        self._fields={}
        line=self.readline()
        line=self.readline().strip()
        i=0
        while line :
            elems = re.split("[ =]+",line)
            self._fields[i] = dict(zip(self.fieldkeys,[el.strip() for el in elems]))
            for k in self.fieldkeys :
                if k in ["min","max","dens","day"] :
                    self._fields[i][k] = float(self._fields[i][k])
                elif k in ["k","step"] :
                    self._fields[i][k] = int(self._fields[i][k])
            i+=1
            line=self.readline().strip()

    def read_field(self,fieldname,level) :
        """ Read field corresponding to fieldname and level from archive file"""
        record = None
        for i,d in self._fields.items() :
            if d["field"] == fieldname and level == d["k"] :
                record=i
        if record  is not None :
            w = self._filea.read_record(record) 
        else :
            w = None
        return w

    def write_header(self) :
        self._fileb.write(self._cline1+"\n")
        self._fileb.write(self._cline2+"\n")
        self._fileb.write(self._cline3+"\n")
        self._fileb.write("12345678901234567890123456789012345678901234567890123456789012345678901234567890\n")
        self.writeitem("iversn",self._iversn)
        self.writeitem("iexpt" ,self._iexpt)
        self.writeitem("yrflag",self._yrflag)
        self.writeitem("idm"    ,self._idm)
        self.writeitem("jdm"    ,self._jdm)
        self._fileb.write("field         time step  model day  k  dens          min                  max\n")

    def write_field(self,field,mask,fieldname,time_step,model_day,k,dens) :
        self._open_filea_if_necessary(field)
        if self._firstwrite :
            self._jdm,self._idm=field.shape
            self._firstwrite=False
            self.write_header()
        self.check_dimensions(field)
        hmin,hmax = self._filea.writerecord(field,mask)
        fmtstr="%-9s=%11d%11.2f%3d%7.3f%16.7E%16.7E\n"
        self._fileb.write(fmtstr%(fieldname,time_step,model_day,k,dens,hmin,hmax))

# Mostafa
    def get_fields(self):
        return self._fields

    @property
    def fieldlevels(self) :
        return set([elem["k"] for elem in self._fields.values()])

    def bminmax(self,fieldname,k) :
        record=None
        for i,d in self._fields.items() :
            if d["field"] == fieldname and d["k"] == k:
                record=i
        if record  is not None :
            ret = (self._fields[i]["min"],self._fields[i]["max"])
        else :
            ret = (None,None)
        return ret

    @property
    def iversn(self) : return self._iversn

    @property
    def iexpt(self) : return self._iexpt

    @property
    def yrflag(self) : return self._yrflag

class ABFileForcing(ABFile) :
    """ Class for doing input/output on pairs of hycom .a and .b files. This is for forcing files"""
    fieldkeys=["field","min","max"]
    def __init__(self,basename,action,mask=False,real4=True,endian="big", idm=None,jdm=None,
                     cline1="",cline2=""):

        super(ABFileForcing,self).__init__(basename,action,mask=mask,real4=real4,endian=endian)
        self._cline1=cline1
        self._cline2=cline2
        if action == "w" :
            pass
        else :
            self.read_header()
            self.read_field_info()
            self._open_filea_if_necessary(numpy.zeros((self._jdm,self._idm)))

    def read_header(self) :
        self._header=[]
        self._header.append(self.readline())
        self._header.append(self.readline())
        self._header.append(self.readline())
        self._header.append(self.readline())
        self._header.append(self.readline())
        self._cline1=self._header[0].strip()
        self._cline2=self._header[1].strip()
        m = re.match("i/jdm[ ]*=[ ]*([0-9]+)[ ]+([0-9]+)",self._header[4].strip())
        if m :
            self._idm = int(m.group(1))
            self._jdm = int(m.group(2))
        else :
            raise  AFileError("Unable to parse idm, jdm from header. File=%s, Parseable string=%s"%(
                    self._filename, self._header[4].strip()))

    def write_header(self) :
        self._fileb.write(self._cline1.strip()+"\n")
        self._fileb.write(self._cline2.strip()+"\n")
        self._fileb.write("\n")
        self._fileb.write("\n")
        self._fileb.write("i/jdm =%5d %5d\n"%(self._idm,self._jdm))

    def write_field(self,field,mask,fieldname,dtime1,rdtime) :
        self._open_filea_if_necessary(field)
        if self._firstwrite :
            self._jdm,self._idm=field.shape
            self._firstwrite=False
            self.write_header()
        self.check_dimensions(field)
        hmin,hmax = self._filea.writerecord(field,mask)
        self._fileb.write("%s:dtime1,range = %12.4f%12.4f,%14.6e%14.6e\n"%(fieldname,dtime1,rdtime,hmin,hmax))

    def read_field_info(self) :
        # Get list of fields from .b file
        #plon:  min,max =        -179.99806         179.99998
        #plat:  min,max =         -15.79576          89.98227
        #...
        self._fields={}
        line=self.readline().strip()
        i=0
        while line :
            m = re.match("^(.*):dtime1,range[ ]*=[ ]+([0-9\-\.e+]+)[ ]+([0-9\-\.e+]+)[ ]*,[ ]*([0-9\-\.e+]+)[ ]*([0-9\-\.e+]+)",line)
            if m :
                self._fields[i] = {}
                self._fields[i]["field"]  = m.group(1).strip()
                self._fields[i]["dtime1"] = float(m.group(2).strip())
                self._fields[i]["range"]  = float(m.group(3).strip())
                self._fields[i]["min"]  = float(m.group(4).strip())
                self._fields[i]["max"]  = float(m.group(5).strip())
            else :
                raise NameError("cant parse forcing field")
            i+=1
            line=self.readline().strip()

    def read_field(self,field,dtime1) :
        """ Read field corresponding to fieldname and level from archive file"""
        elems = [ (k,v["dtime1"]) for k,v in self._fields.items() if v["field"] == field]
        dist = numpy.array([elem[1]-dtime1 for elem in elems])
        i =numpy.argmin(numpy.abs(dist))
        rec,dt = elems[i]
        w = self._filea.read_record(i) 
        #print w
        return w#,dt

    def bminmax(self,fieldname,dtime1) :
        record=None
        for i,d in self._fields.items() :
            if d["field"] == fieldname and d["dtime1"] == dtime1:
                record=i
        if record  is not None :
            ret = (self._fields[i]["min"],self._fields[i]["max"])
        else :
            ret = (None,None)
        return ret

    @property
    def field_times(self) :
        return set([elem["dtime1"] for elem in self._fields.values()])

class ABFileRiver(ABFile) :
    """ Class for doing input/output on pairs of hycom .a and .b files. This is for river forcing files"""
    """ At the moment, only write river function is implemented. You can read it by using 'loada' subroutines """
    fieldkeys=["field","min","max"]
    def __init__(self,basename,action,mask=False,real4=True,endian="big", idm=None,jdm=None,
                     cline1="",cline2=""):

        super(ABFileRiver,self).__init__(basename,action,mask=mask,real4=real4,endian=endian)
        self._idm=idm
        self._jdm=jdm
        self._cline1=cline1
        self._cline2=cline2
        if action == "w" :
            pass
        else :
            quit()

    def write_header(self) :
        self._fileb.write(self._cline1.strip()+"\n")
        self._fileb.write(self._cline2.strip()+"\n")
        self._fileb.write("\n")
        self._fileb.write("\n")
        self._fileb.write("i/jdm =%5d %5d\n"%(self._idm,self._jdm))

    def write_field(self,field,mask,fieldname,month) :
        self._open_filea_if_necessary(field)
        self.check_dimensions(field)
        hmin,hmax = self._filea.writerecord(field,mask)
        self._fileb.write(" %s: month,range = %s%14.6E%14.6E\n"%(fieldname,str(month).zfill(2),hmin,hmax))

class ABFileRestart(ABFile) :
    """ Class for doing input/output on pairs of hycom .a and .b files. This is for restart files"""
    fieldkeys=["field","step","day","k","dens","min","max"]
    def __init__(self,basename,action,mask=False,real4=True,endian="big",
            iversn=None,iexpt=None,yrflag=None,idm=None,jdm=None) :

        super(ABFileRestart,self).__init__(basename,action,mask=mask,real4=real4,endian=endian)
        if self._action[0] == "r" :
            if idm != None and jdm != None:
                self._idm=idm
                self._jdm=jdm
                self.read_header() # Sets internal metadata. Overrides those on input
                self.read_field_info()
                self._open_filea_if_necessary(numpy.zeros((self._jdm,self._idm)))
            else :
                raise BFileError("ABFile opened as read, but idm and jdm not provided")
        elif self._action == "w" :
##            # Need to test if idm, jdm, etc is set at this stage
##            raise NotImplementedError,"ABFileRestart writing not implemented
            pass

    def read_header(self) :
        #RESTART2: iexpt,iversn,yrflag,sigver =     990     22      3      2
        #RESTART2: nstep,dtime,thbase =        19636860     40910.12500000000            34.00000000000000
        self._header=[]
        self._header.append(self.readline())
        self._header.append(self.readline())

        m=re.match("RESTART2: iexpt,iversn,yrflag,sigver[ ]*=[ ]*([0-9]+)[ ]+([0-9]+)[ ]+([0-9]+)[ ]+([0-9]+)",self._header[0])
        self._iexpt=int(m.group(1))
        self._iversn=int(m.group(2))
        self._yrflag=int(m.group(3))
        self._sigver=int(m.group(4))
        m2=re.match("RESTART2: nstep,dtime,thbase[ ]*=[ ]*([^ ]+)[ ]+([^ ]+)[ ]+([^ ]+)",self._header[1])
        self._nstep=int(m2.group(1))
        self._dtime=float(m2.group(2))
        self._thbase=float(m2.group(3))

    def read_field_info(self) :
        #u         : layer,tlevel,range =    1  1     -9.2374402E-01    6.0595530E-01
        self._fields={}
        line=self.readline().strip()
        i=0
        while line :
            m = re.match("^([a-zA-Z0-9_ ]+): layer,tlevel,range =[ ]*([^ ]+)[ ]+([^ ]+)[ ]+([^ ]+)[ ]+([^ ]+)",line)
            if m :
                self._fields[i] = {}
                self._fields[i]["field"] = m.group(1).strip()
                self._fields[i]["k"] = int(m.group(2))
                self._fields[i]["tlevel"] = int(m.group(3))
                self._fields[i]["min"] = float(m.group(4))
                self._fields[i]["max"] = float(m.group(5))
            else :
                raise NameError("unable to parse line %s"%line)
            #print i,line
            i+=1
            line=self.readline().strip()

    def write_bfile_info(self) :
        self._fileb.seek(0)
        ##header
        self._fileb.write("RESTART2: iexpt,iversn,yrflag,sigver = %6d%6d%6d%6d\n"%(self._iexpt,self._iversn,self._yrflag,self._sigver))
        self._fileb.write("RESTART2: nstep,dtime,thbase = %12d%19.10f%24.13f\n"%(self._nstep,self._dtime,self._thbase))
        ##field info
        for i in range(len(self._fields)):
            fieldname = self._fields[i]["field"]
            k = self._fields[i]["k"]
            time = self._fields[i]["tlevel"]
            hmin = self._fields[i]["min"]
            hmax = self._fields[i]["max"]
            fmtstr="%-8s: layer,tlevel,range = %3d%3d%18.7E%16.7E\n"
            self._fileb.write(fmtstr%(fieldname,k,time,hmin,hmax))

    def find_record(self, fieldname, level, tlevel):
        record = None
        for i,d in self._fields.items() :
            if d["field"] == fieldname and level == d["k"] and d["tlevel"] == tlevel :
                record=i
        return record

    def read_field(self,fieldname,level,tlevel=1,mask=None) :
        """ Read field corresponding to fieldname and level from archive file"""
        record = self.find_record(fieldname, level, tlevel)
        if record  is not None :
            w = self._filea.read_record(record)
            ABFile.check_minmax(w,self._fields[record]) # Always do this check
            fld = w.data
            if mask is None:
                fld[w.mask] = numpy.nan
            else:
                fld[mask] = numpy.nan
        else :
            fld = None
        return fld

    def overwrite_field(self, field, mask, fieldname, level, tlevel=1) :
        assert self._action == "r+"
        record = self.find_record(fieldname, level, tlevel)
        assert record is not None, "Do not find record in current file, existing"
        self._open_filea_if_necessary(field)
        self.check_dimensions(field)
        hmin,hmax = self._filea.writerecord(field, mask, record=record)
        self._fields[record]["min"] = hmin
        self._fields[record]["max"] = hmax
        self.write_bfile_info()

# THE RESTART WRITING - a later addition to KAL's original abfile.py
#  caglar: other classes in abfile.py code call 'write_header' within 'write_field'.
#             Because of many inputs in the header, I decided to do it separately.
#             Make sure that you call 'write_header' only once before calling the rest of 'write_field'
#  example lines:
#  newfile = 'merged'
#  new_abfile = abfile.ABFileRestart(newfile,"w",idm=I2,jdm=I1)
#  new_abfile.write_header(iexpt,iversn,yrflag,sigver,nstep,dtime,thbase)
#  ~~ for loop for variables
#      new_abfile.write_field(field,None,fieldname,k,t)  # None stands for mask
#  new_abfile.close()
##YY: the above is suitable for writing a new file from scratch (copy and paste)
##overwrite_field will be good for rewriting only a field in existing files

    def write_header(self,iexpt,iversn,yrflag,sigver,nstep,dtime,thbase) :
        self._fileb.write("RESTART2: iexpt,iversn,yrflag,sigver = %6d%6d%6d%6d\n"%(iexpt,iversn,yrflag,sigver))
        self._fileb.write("RESTART2: nstep,dtime,thbase = %12d%19.10f%24.13f\n"%(nstep,dtime,thbase))

    def write_field(self,field,mask,fieldname,k,time) :
        self._open_filea_if_necessary(field)
        self.check_dimensions(field)
        hmin,hmax = self._filea.writerecord(field,mask)
        fmtstr="%-8s: layer,tlevel,range = %3d%3d%18.7E%16.7E\n"
        self._fileb.write(fmtstr%(fieldname,k,time,hmin,hmax))

    @property
    def fieldlevels(self) :
        return set([elem["k"] for elem in self._fields.values()])

    def bminmax(self,fieldname,k) :
        record=None
        for i,d in self._fields.items() :
            if d["field"] == fieldname and d["k"] == k:
                record=i
        if record  is not None :
            ret = (self._fields[i]["min"],self._fields[i]["max"])
        else :
            ret = (None,None)
        return ret

class ABFileRelax(ABFile) :
    """ Class for doing input/output on pairs of hycom .a and .b files. This is for hybrid coord relaxation data used by hycom """
    fieldkeys=["field","layer","dens","min","max"]
    def __init__(self,basename,action,mask=False,real4=True,endian="big", idm=None,jdm=None,
                     cline1="",cline2=""):
        super(ABFileRelax,self).__init__(basename,action,mask=mask,real4=real4,endian=endian)
        self._cline1=cline1
        self._cline2=cline2
        if action == "w" :
            raise NotImplementedError
            pass
        else :
            self.read_header()
            self.read_field_info()
            self._open_filea_if_necessary(numpy.zeros((self._jdm,self._idm)))

#woa2013 Climatology
#Expt 99.0  nhybrd=32 nsigma=14 ds00= 0.50 dp00= 3.00 dp00x= 450.0 dp00f=1.180
#Layered averages w.r.t. Sigma-2,  levtop=1 (sigver= 2)
#Potential Temperature
#i/jdm =  800  760
    def read_header(self) :
        self._header=[]
        self._header.append(self.readline())
        self._header.append(self.readline())
        self._header.append(self.readline())
        self._header.append(self.readline())
        self._header.append(self.readline())
        #print "test",self._header
        m = re.match("i/jdm[ ]*=[ ]*([0-9]+)[ ]+([0-9]+)",self._header[4].strip())
        if m :
            self._idm = int(m.group(1))
            self._jdm = int(m.group(2))
        else :
            raise  BFileError("Unable to parse idm, jdm from header. File=%s, Parseable string=%s"%(
                    self._basename, self._header[4].strip()))
        #print self._idm,self._jdm

    def read_field_info(self) :
        # Typical line 
        #tem: month,layer,dens,range = 01  01 28.100  -3.3520899E+00    1.0784083E+01
        self._fields={}
        line=self.readline().strip()
        i=0
        while line :
            m = re.match("^([a-z_ ]+):[ ]*month[ ]*,[ ]*layer[ ]*,[ ]*dens[ ]*,[ ]*range[ ]*=[ ]*([^ ]+)[ ]+([^ ]+)[ ]+([^ ]+)[ ]+([^ ]+)[ ]+([^ ]+)",line)
            if m :
                self._fields[i] = {}
                self._fields[i]["field"] = m.group(1).strip()
                self._fields[i]["month"] = int(m.group(2))
                self._fields[i]["k"] = int(m.group(3))
                self._fields[i]["dens"] = float(m.group(4))
                self._fields[i]["min"] = float(m.group(5))
                self._fields[i]["max"] = float(m.group(6))
                #print self._fields[i]
            else :
                raise NameError("unable to parse line %s"%line)
            #print i,line
            i+=1
            line=self.readline().strip()

    def read_field(self,fieldname,level,month) :
        """ Read field corresponding to fieldname and level from archive file"""
        record = None
        for i,d in self._fields.items() :
            if d["field"] == fieldname and level == d["k"] and d["month"] == month :
                record=i
        if record  is not None :
            w = self._filea.read_record(record) 
            ABFile.check_minmax(w,self._fields[record]) # Always do this check
        else :
            w = None
        return w

class ABFileRelaxZ(ABFile) :
    """ Class for doing input/output on pairs of hycom .a and .b files. This is for z level data used by hycom relax routine """
    fieldkeys=["field","depth","min","max"]
    def __init__(self,basename,action,mask=False,real4=True,endian="big", idm=None,jdm=None,
                     cline1="",cline2=""):
        super(ABFileRelaxZ,self).__init__(basename,action,mask=mask,real4=real4,endian=endian)
        self._cline1=cline1
        self._cline2=cline2
        if action == "w" :
            pass
        else :
            self.read_header()
            self.read_field_info()
            self._open_filea_if_necessary(numpy.zeros((self._jdm,self._idm)))

    def read_header(self) :
        self._header=[]
        self._header.append(self.readline())
        self._header.append(self.readline())
        self._header.append(self.readline())
        self._header.append(self.readline())
        self._header.append(self.readline())
        self._cline1=self._header[0].strip()
        self._cline2=self._header[1].strip()
        m = re.match("i/jdm[ ]*=[ ]*([0-9]+)[ ]+([0-9]+)",self._header[4].strip())
        if m :
            self._idm = int(m.group(1))
            self._jdm = int(m.group(2))
        else :
            raise  AFileError("Unable to parse idm, jdm from header. File=%s, Parseable string=%s"%(
                    self._filename, self._header[4].strip()))

    def write_header(self) :
        self._fileb.write(self._cline1.strip()+"\n")
        self._fileb.write(self._cline2.strip()+"\n")
        self._fileb.write("\n")
        self._fileb.write("\n")
        self._fileb.write("i/jdm =%5d %5d\n"%(self._idm,self._jdm))

    def write_field(self,field,mask,fieldname,depth) :
        self._open_filea_if_necessary(field)
        if self._firstwrite :
            self._jdm,self._idm=field.shape
            self._firstwrite=False
            self.write_header()
        self.check_dimensions(field)
        hmin,hmax = self._filea.writerecord(field,mask)
        self._fileb.write("%s: depth,range = %12.4f %14.6e%14.6e\n"%(fieldname,depth,hmin,hmax))

    def read_field_info(self) :
        # Get list of fields from .b file
        #plon:  min,max =        -179.99806         179.99998
        #plat:  min,max =         -15.79576          89.98227
        #...
        self._fields={}
        line=self.readline().strip()
        i=0
        while line :
            m = re.match("^(.*):[ ]*depth,[ ]*range[ ]*=[ ]*([0-9\-\.e+]+)[ ]+([0-9\-\.e+]+)[ ]+([0-9\-\.e+]+)",line)
            if m :
                self._fields[i] = {}
                self._fields[i]["field"]  = m.group(1).strip()
                self._fields[i]["depth"] = float(m.group(2).strip())
                self._fields[i]["min"]  = float(m.group(3).strip())
                self._fields[i]["max"]  = float(m.group(4).strip())
            else :
                raise NameError("cant parse forcing field")
            i+=1
            line=self.readline().strip()

    def read_field(self,field,dtime1) :
        """ Read field corresponding to fieldname and level from archive file"""
        elems = [ (k,v["dtime1"]) for k,v in self._fields.items() if v["field"] == field]
        dist = numpy.array([elem[1]-dtime1 for elem in elems])
        i =numpy.argmin(numpy.abs(dist))
        rec,dt = elems[i]
        w = self._filea.read_record(i) 
        #print w
        return w#,dt

    def bminmax(self,fieldname,depth) :
        record=None
        for i,d in self._fields.items() :
            if d["field"] == fieldname and d["depth"] == dtime1:
                record=i
        if record  is not None :
            ret = (self._fields[i]["min"],self._fields[i]["max"])
        else :
            ret = (None,None)
        return ret

def write_bathymetry(exp,version,d,threshold) :
    myfile="depth_%s_%02d"%(exp,version)
    #regf = ABFileBathy(myfile,"w",idm=d.shape[0],jdm=d.shape[1],mask=True)
    #print type(d),d.min(),d.max()
    #mask=d <= threshold
    #regf.write_field(d,mask)
    #No mask!
    tmp = numpy.copy(d)
    mask = tmp <= threshold
    tmp[mask] = AFile.huge
    regf = ABFileBathy(myfile,"w",idm=d.shape[0],jdm=d.shape[1],mask=True)
    regf.write_field(tmp,mask)
    regf.close()

def write_regional_grid(datadict,endian="big") :
    regf = ABFileGrid("regional.grid","w",mapflg=-1,endian=endian)
    for key in grid_ordered_fieldnames :
        regf.write_field(datadict[key],datadict[key],key)
    regf.close()

def read_regional_grid(endian="big") :
    regf = ABFileGrid("regional.grid","r",endian=endian)
    res={}
    for fldname in grid_ordered_fieldnames :
        res[fldname] = regf.read_field(fldname)
    regf.close()
    return res

def write_diag_nc(datadict,fname="hycom_grid.nc") :
    try :
        import netCDF4
    except :
        print("Unable to import netCDF4 module - will not write diag to %s"%fname)
        return
    tmp = datadict["plon"]
    ## Create netcdf file with all  stages for analysis
    ncid = netCDF4.Dataset(fname,"w")
    ncid.createDimension("idm",tmp.shape[1])
    ncid.createDimension("jdm",tmp.shape[0])
    for i in datadict.keys() :
        ncid.createVariable(i,"f8",("jdm","idm"))
    for i in datadict.keys() :
        ncid.variables[i][:]=datadict[i][:]
    ncid.close()

