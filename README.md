## deepSI\_lite

deepSI\_lite provides a lightweight pytorch based framework for data-driven learning of dynamical systems (i.e. system identification). It contains a large forcus on the SUBNET method which is able to robustly model many systems.

## Features

* A number of popular SUBNET structures
  * SUBNET encoder structue (`deepSI_lite.models.SUBNET`)
    * see: https://arxiv.org/abs/2012.07697 or https://www.sciencedirect.com/science/article/pii/S0005109823003710
  * Continuous time SUBNET encoder structure (`deepSI_lite.models.SUBNET_CT`)
    * see: https://arxiv.org/abs/2204.09405 
  * Base class for fully custom SUBNET structures with shared parameters between `f`, `h` or `encoder`. (`deepSI_lite.models.Custom_SUBNET`)
  * CNN SUBNET (`CNN_SUBNET`)
    * see chapter 4: https://research.tue.nl/files/318935789/20240321_Beintema_hf.pdf
  * LPV SUBNET (`SUBNET_LPV` and `SUBNET_LPV_ext_scheduled`)
    * see: https://arxiv.org/abs/2204.04060
  * HNN SUBNET (`pHNN_SUBNET`)
    * see: https://arxiv.org/abs/2305.01338
* Connection to [`nonlinear_benchmarks`](https://github.com/GerbenBeintema/nonlinear_benchmarks) such that benchmarks can easily be loaded and evaluated on.
* Low amount of code such that it can be easily forked and edited to add missing features.

## Installation

```
conda install -c anaconda git
pip install git+https://github.com/GerbenBeintema/deepSI_lite@main
```

## Example usage

```python
import nonlinear_benchmarks as nlb
import deepSI_lite as dsi

# get benchmark data:
train_val, test = nlb.WienerHammerBenchMark()
split_id = int(len(train_val)*0.99)
train, val = train_val[:split_id], train_val[split_id:] #split train and val
# alternative:
#train_val = dsi.Input_output_data(u=..., y=..., sampling_time=...)

# charaterize data (number of input nu, number of output ny and the norm)
nu, ny, norm = dsi.get_nu_ny_and_auto_norm(train_val)

# create a SUBNET model
model = dsi.SUBNET(nu, ny, norm=norm, nx=6, nb=40, na=40) #MLP 2 hidden layer for all three components

# optimize using NRMSE and stochastic gradient descent (Adam).
train_dict = dsi.fit(model, train, val, n_its=10_000, T=80, batch_size=128, val_freq=100) #only 10K iterations for example (1M closer to optimal)

#simulate the model
test_p = model.simulate(test) 
k0 = max(model.na, model.nb) #split off the encoder lenght
print(f'RMS = {((test.y[k0:] - test_p.y[k0:])**2).mean()**0.5:.5f} V ')
```

## Futher documentation

Check out `examples/Demonstration deepSI_lite.ipynb`.

## todo list and known issues

* Known issues: The compile option in `fit` currently has a memory leak??
* Expand demonstration notebook with pHNN examples
* LPV SUBNET with exernal schedualing.
* General documentation 
* known issues: CT SUBNET and DT SUBNET does not produce the correct initial when the sampling time is altered. (the encoder assumes that the sampling time does not change)
* Streamline the user experiance when a model only has been implemented for MIMO and a SISO dataset is given.
* Change name from `deepSI_lite` -> `deepSI` 
* pypi data upload such that it can be easily installed with `pip install deepSI_lite`
* Improve speed with copy if enough memory is available. Also pre-transfer to GPU and maybe asyncroness getting of arrays.
