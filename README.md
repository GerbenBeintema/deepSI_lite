## deepSI\_lite

deepSI\_lite a lightweight and flexible implementation of the SUBNET structure for data-driven modeling of dynamical systems (i.e. system identification). 

## Features

* A number of popular SUBNET structures
  * SUBNET encoder structue (`deepSI_lite.models.SUBNET`)
  * Continuous time SUBNET encoder structure (`deepSI_lite.models.SUBNET_CT`)
  * Base class for fully custom SUBNET structures with shared parameters between `f`, `h` or `encoder`. (`deepSI_lite.models.Custom_SUBNET`)
  * CNN SUBNET (`CNN_SUBNET`)
  * LPV SUBNET (`SUBNET_LPV`)
  * HNN SUBNET (`HNN_SUBNET`)
* Connection to `nonlinear_benchmarks` such that benchmarks can easily be loaded and evaluated on.
* Low amount of code such that it can be easily forked and edited to add missing features.

## Installation

```
conda install -c anaconda git
pip install git+https://github.com/GerbenBeintema/deepSI_lite@master
```

## Example usage

```python
import deepSI_lite
```

## Futher documentation

Check out `examples/Demonstration deepSI_lite.ipynb`.

## todo list

* The compile option in `fit` currently has a memory leak?
* Expand demonstration notebook
* General documentation 
* BUG/missing feature: CT SUBNET and DT SUBNET does not produce the correct initial when the sampling time is altered. (the encoder assumes that the sampling time does not change)
* Streamline the user experiance when a model only has been implemented for MIMO and a SISO dataset is given.
* Streamline the ONNX exporting such that the estimated models can be easily loaded in for instance in MATLAB
* Change name from `deepSI` -> `deepSI_lite`
* pypi data upload such that it can be easily installed with `pip install deepSI_lite`
* Regularization example in demonstration notebook
