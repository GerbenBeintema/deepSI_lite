## deepSI\_lite

deepSI\_lite a lightweight and flexible implementation of the SUBNET structure for data-driven modeling of dynamical systems (i.e. system identification). 

## Features

* Model structure
  * SUBNET encoder structue `deepSI_lite.models`
  * Continuous time SUBNET encoder structure (`deepSI_lite.models_CT`)
  * Base class for fully custom SUBNET structures with shared parameters between `f`, `h` or `encoder`. (`deepSI_lite.Custom_SUBNET`)

## Installation


```
conda install -c anaconda git
pip install git+https://github.com/GerbenBeintema/deepSI_lite@master
```

## todo list

* Compile? -> Memory Leak?
* Create Example notebook
  * Basic usage
* CT with sample_time = None? -> That's not allows
  * CT with list of data currently does not work
* multi variate ny=int and norm  
* Validate implementation with re-running of old results
* Exporting to MATLAB
* General documentation 
* pip install deepSI
* Change name from deepSI -> deepSI_lite
* warn upon accidental mixing of sample times
* changing of sample times with encoder (re-sample)
* DT SUBNET should check if the sample time is unchanged!
* Model estimate validation
* Export
* Regularization
* nu scalar -> nu=1, Auto conversion to MIMO 
* CT shared parameters
* Multiple aux from forward.
* Aux loss function inputs

