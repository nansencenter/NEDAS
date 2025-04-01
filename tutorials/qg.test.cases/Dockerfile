##get Intel Fortran compiler
FROM intel/fortran-essentials:latest
ENV PATH="/opt/intel/oneapi/compiler/latest/linux/bin/intel64:$PATH"
ENV LD_LIBRARY_PATH="/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin:$LD_LIBRARY_PATH"

##get FFTW3
RUN apt-get update && apt-get install -y libfftw3-dev

##get NEDAS and required libraries
RUN git clone https://github.com/nansencenter/NEDAS.git /NEDAS
WORKDIR /NEDAS
RUN git checkout -b develop origin/develop
RUN pip install -r requirements.txt

##compile qg model
WORKDIR /NEDAS/models/qg/src
RUN make

##it is necessary to relax the stack limit for qg model to run
RUN ulimit -s 2000000

WORKDIR /NEDAS

#CMD [ "python", "scripts/run_expt.py", "-c", "config/samples/qg.yml" ]
