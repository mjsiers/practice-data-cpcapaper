#%%
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd 
from scipy.stats import norm    

mpl.style.use('seaborn-notebook')
plt.rcParams["figure.figsize"] = (12, 5)

#%%
xnum = 600
np.random.seed(42)
x = np.arange(0, xnum, 1.0)
E1_ = (-1e-7*x**1.8)
E1F_ = E1_ + np.min(E1_)*-1.0
E2_ = (-1e-7*x**2.0)
E2F_ = E2_ + np.min(E2_)*-1.0

# bnoise = 0.00100 * np.random.normal(size=xnum)

fig, axs = plt.subplots()
axs.plot(x, E1F_, label='min')
axs.plot(x, E2F_, label='max')
fig.suptitle('Background 1 Base Shapes')
plt.legend()


#%%
S_1 = norm.pdf(x, loc=310.0, scale=40.0)
S_2 = norm.pdf(x, loc=390.0, scale=20.0)
S_3 = norm.pdf(x, loc=360.0, scale=120.0)

fig, axs = plt.subplots()
axs.plot(x, S_1)
axs.plot(x, S_2)
axs.plot(x, S_3)
fig.suptitle('Signal Base Gaussian Shapes')
