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

# %% [markdown]
# rename paths from unix timestamp to iso 8601

# %%
"""
import os
from pathlib import Path
basedir = Path("data")
for fn in basedir.glob('*.zip'):
    
    file_split = fn.name.split("_")
    if len(file_split) != 2:
        continue
    try:
        file_split[0] = datetime.utcfromtimestamp(float(file_split[0])).isoformat()
    except Exception as e:
        continue
    new_file = basedir / "_".join(file_split)
    fn.rename(new_file)
"""

# %%
"""
import os
from pathlib import Path
basedir = Path("data")
for fn in os.listdir(basedir):
    fn = basedir / fn
    if not fn.is_dir():
        continue
    
    file_split = fn.name.split("_")
    if len(file_split) != 2:
        continue
    try:
        file_split[0] = datetime.utcfromtimestamp(float(file_split[0])).isoformat()
    except Exception as e:
        continue
    new_file = basedir / "_".join(file_split)
    fn.rename(new_file)
"""

# %% [markdown]
# # Task 3
# ## Imports

# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbn#

import zipfile
from datetime import datetime


# %% [markdown]
# ## Reading Data

# %%
def _read_zipped_csv(archive, path):
    with archive.open(path,"r") as f:
        df = pd.read_csv(f,header=None)
        timestamp = datetime.utcfromtimestamp(float(df.iloc[0,0]))
        sample_rate = float(df.iloc[1,0])
        df = df.iloc[2:]
        if len(df.columns) == 1:
            df = np.array(df[0])
    return {"timestamp":timestamp
            ,"sample_rate":sample_rate
            ,"data": df}

def read_e4_data(path):
    sensors = ["TEMP", "EDA", "BVP", "ACC", "HR"]#"IBI",
    annotations = ["tags"]
    data = {}
    with zipfile.ZipFile(path, 'r') as archive:
        for s in sensors:
            file_name = f"{s}.csv"
            #print(file_name)
            data[s] = _read_zipped_csv(archive, file_name)
    return data


# %%
contr_1 = read_e4_data("data/2019-10-21T14:21:47_A0208E.zip")
contr_2 = read_e4_data("data/2019-10-21T14:21:48_A023B2.zip")

exp_1   = read_e4_data("data/2019-10-21T14:32:26_A0208E.zip")
exp_2   = read_e4_data("data/2019-10-21T14:32:27_A023B2.zip")

# %% [markdown]
# ## Bonus Hypothesis #1 (ANOVA)
#
# Hypothesis: **Hypothesis: Participants in the experimental condition collaborate more, using more body language and moving their hands more, than participants in the control group.**

# %%
ACC_SAMPLE_RATE = int(exp_1["ACC"]["sample_rate"])
def acc_abs(data):
    tmp = np.linalg.norm(data["ACC"]["data"],axis = 1)
    #tmp -= np.mean(tmp)
    return tmp

#only look at last 180ms
def region_of_interest(data):
    return list(data[-200*ACC_SAMPLE_RATE:-20*ACC_SAMPLE_RATE])

e1_acc = acc_abs(exp_1)
e2_acc = acc_abs(exp_2)

c1_acc = acc_abs(contr_1)
c2_acc = acc_abs(contr_2)

from statsmodels.formula.api import ols
groups = np.array(list([0]*ACC_SAMPLE_RATE*180*2) + list([1]*ACC_SAMPLE_RATE*180*2))

data_exp = region_of_interest(e1_acc)+ region_of_interest(e2_acc)
data_controll = region_of_interest(c1_acc)+region_of_interest(c2_acc)

all_data= data_exp + data_controll
all_data = np.array(all_data)
print(np.mean(data_exp))
print(np.mean(data_controll))

df = pd.DataFrame(np.array([all_data,groups]).T,columns=["data","group"])
results = ols('data ~ C(group)', data=df).fit()
results.summary()

# %% [markdown]
# Result: The anova shows an overall significant difference ($p = 7.10*10^{-58}$).
# Since we only had 2 groups (experiment vs. controll) the anova shows a significant difference between 
# the mean absolute acceleration between these two groups. This might indicate that participants in the experimental group were more creative and used more body language or it might be just by chance, since the sample size is so small.

# %%
plt.hist(data_exp)
plt.hist(data_controll)

# %% [markdown]
# ## Bonus Hypothesis 2 (LinReg)
#
# Hypothesis: Participants in the experimental group experience less stress and therefore have a lower heartrate than participants in the controll group.

# %%
plt.plot(exp_1["HR"]["data"])
plt.plot(exp_2["HR"]["data"])
plt.plot(contr_1["HR"]["data"])
plt.plot(contr_2["HR"]["data"])
plt.legend(["e1","e2","c1","c2"])

# %%
hr_exp = list(exp_1["HR"]["data"][10:]) + list(exp_2["HR"]["data"][10:])
hr_contr = list(contr_1["HR"]["data"][10:]) + list(contr_2["HR"]["data"][10:])

y = [1] * len(hr_exp) + [0]*len(hr_contr)

print(np.mean(contr_1["HR"]["data"][-180:]))
print(np.mean(exp_1["HR"]["data"][-180:]))

regressor = LinearRegression()  
regressor.fit(np.array([y]).T,hr_exp+hr_contr)
pred = regressor.predict([[0],[1]])
plt.plot(pred)

# %% [markdown]
# Result: The average heartrate in the control group was 91 and 94 in the experimental condition group. Contrary to our expectations, participants in the experimental group had a higher mean heartrate. This is probably just by chance, since we only had 2 participants in each group and did an between subjects study. Additionally, no baseline heartrate was measured, so it is not clear whether the higher heart rate was due to the activity, or whether the participants simply had higher resting heart rates. The heart rate of healthy humans at rest can range from around 50 to 100 bpm.
# The largest problem we faced when analysing the data, was not being able to sync the data with the videos, thus not knowing when the actual experiment started and ended. We also did not record baseline measurements before or after the experiment as would be needed for EDA analysis.

# %%
#plt.plot(e1_acc[-180*ACC_SAMPLE_RATE:])
"""
plt.hist(region_of_interest(e2_acc))
plt.hist(region_of_interest(e1_acc))

plt.hist(region_of_interest(c1_acc))
plt.hist(region_of_interest(c2_acc))
#plt.hist(c1_acc[-180*ACC_SAMPLE_RATE:])
#plt.hist(c2_acc[-180*ACC_SAMPLE_RATE:])
"""

# %%
