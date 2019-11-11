# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn

# %%
data_after = pd.read_csv("NeuroDesignVid02.csv")
data_before = pd.read_csv("NeuroDesignVid01.csv")

# %%
action_units = [x for x in data.columns if "AU" in x and "_r" in x]

# %%
plt.plot(data[" AU06_r"])
plt.plot(data[" AU12_r"])

# %%
np.mean(data_before[" AU06_r"] + data_before[" AU12_r"])

# %%
np.mean(data_after[" AU06_r"] + data_after[" AU12_r"])

# %%
sbn.distplot(data_before[" AU06_r"] + data_before[" AU12_r"])

# %%
sbn.distplot( data_after[" AU06_r"] + data_after[" AU12_r"])

# %%
