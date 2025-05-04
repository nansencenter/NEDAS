<img src='/docs/imgs/nedas_logo.png' width='250' align='left' style='padding-right:20px'/>

The Next-generation Ensemble Data Assimilation System (NEDAS) is a lightweight, modular Python platform
designed for rapid prototyping and testing of data assimilation (DA) algorithms in geophysical models.
Leveraging [mpi4py](https://mpi4py.readthedocs.io/en/stable/), [numpy](https://numpy.org/),
and [numba](https://numba.pydata.org/),
NEDAS ensures scalability and computational efficiency for high-dimensional systems.
Thanks to the rich Python ecosystem for machine learning, NEDAS provides a platform for early-stage development of novel DA methods.
Moreover, NEDAS offers a collection of state-of-the-art DA algorithms for benchmarking,
including the serial assimilation approaches (similar to [DART](https://github.com/NCAR/DART)) and
batch assimilation approaches (similar to the LETKF in [PDAF](https://pdaf.awi.de/trac/wiki)).

[NEDAS documentaion is available online](https://nedas.readthedocs.io/en/latest/)

### Installing NEDAS

NEDAS is available through this Github repository. To install the latest version at `[install path]`:

```
cd [install path]
git clone https://github.com/nansencenter/NEDAS.git 
pip install -r [install path]/NEDAS/requirements.txt
export PYTHONPATH=[install path]/NEDAS:$PYTHONPATH
```

Or, download from the PyPI platform directly:

```
pip install NEDAS
```

See more details in the [installation instructions](https://nedas.readthedocs.io/en/latest/installation.html).

### Acknowledgements

NEDAS was initiated by [Yue Ying](https://myying.github.io/) in 2022. Please cite this repository [![DOI](https://zenodo.org/badge/485034698.svg)](https://zenodo.org/doi/10.5281/zenodo.10525330) if you used NEDAS to produce results in your research publication/presentation.

The developement of this software was supported by the NERSC internal funding in 2022; and the Scale-Aware Sea Ice Project ([SASIP](https://sasip-climate.github.io/)) in 2023-2024.

With contribution from: Anton Korosov, Timothy Williams (pynextsim libraries), NERSC-HYCOM-CICE group led by Annette Samuelsen (pythonlib for abfile, confmap, etc.), Jiping Xie (enkf-topaz), Tsuyoshi Wakamatsu (BIORAN), Francois Counillon, Yiguo Wang, Tarkeshwar Singh (EnOI, EnKF, and offline EnKS in NorCPM).

We provide the software "as is", the user is responsible for their own modification and ultimate interpretation of their research findings using the software. We welcome community feedback and contribution to support new models/observations, please use the "pull request" if you want to be part of the development effort.
