This project is an implementation of the Sea-thru algorithm.

The implementations for the original paper are contained in files `sea_thru.py`, `compute_illuminant.cpp`, and `simple_illuminant.cpp`.

Folder `midas/` contains a fork from repository [isl-org/MiDaS](https://github.com/isl-org/MiDaS) which contains an implementation of paper *Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer* from Intel ISL (please see `LICENSE_MiDaS` for permissions).

`midas_helper.py` is an interface written to call MiDaS from our implementation of Sea-thru.
