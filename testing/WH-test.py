from deepSI_lite.SUBNET import SUBNET
import nonlinear_benchmarks

train_val, test = nonlinear_benchmarks.WienerHammerBenchMark()
s = int(len(train_val)*0.99)
train, val = train_val[:s], train_val[s:]

from deepSI_lite.normalization import get_nu_ny_and_auto_norm
nu, ny, norm = get_nu_ny_and_auto_norm(train_val)

nx = 6
model = SUBNET(nu, ny, norm=norm, nx=nx, nb=40, na=40, f=None, h=None, encoder=None, feedthrough=False) 

from deepSI_lite.fitting import fit
train_dict = fit(model, train, val, n_its=1_000_000, T=80, batch_size=128, stride=1, val_freq=100)

