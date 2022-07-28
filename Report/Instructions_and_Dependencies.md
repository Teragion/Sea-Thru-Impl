Dependencies are those listed in `sea_thru.py` and `midas_helper.py`. Namely, the following python packages:

For Sea-thru:
* `numpy`
* `scikit-image`
* `scikit-learn`
* `scipy`
* `opencv`
* `rawpy`
  - to work on apple silicon Macs, you may need to install `rawpy` from source

For MiDaS integration to work:
* `pytorch`
* `torchvision`
* `timm`

To compile `C++` modules including `compute_illuminant.cpp` and `simple_illuminant.cpp`, use command like 

```
/path/to/clang++ -O2 -o sillu.<OS specific suffix> -shared <src>.cpp -std=c++17 -lomp -fopenmp
```
where `<OS specific suffix>` is something like `.so`, `.dylib`, etc.

The code is implemented and tested under macOS.
