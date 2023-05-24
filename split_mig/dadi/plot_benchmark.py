import glob
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import pandas as pd
from pylab import *

file_list_20 = sorted(glob.glob("ns_20/outfiles/dadi/*out"))
file_list_160 = sorted(glob.glob("ns_160_copy/outfiles/dadi/*out"))


def get_runtime(filename_list):
    runtime_list = []
    array_str = ''
    count=0
    for fname in filename_list:
        file = open(fname, "r")
        time_file = []
        for line in file:
            if 'MST' in line:
                extract_time = datetime.datetime.strptime(
                    line.strip(), '%a %b %d %H:%M:%S %Z %Y')
                time_file.append(extract_time)
        if len(time_file) == 2:
            runtime = time_file[1] - time_file[0]
            # convert to cpu hours
            runtime_in_cpu_hours = runtime.total_seconds()/60.0/60.0*10
            runtime_list.append(runtime_in_cpu_hours)
        else:
            # timed out job, add to the largest bin
            array_str += fname.split('_')[-1].split('.')[0] + ','
            runtime_list.append(501)
            count+=1
    # print(array_str)
    # print(count)
    return runtime_list
    
runtime_list_20 = get_runtime(file_list_20)
runtime_list_160 = get_runtime(file_list_160)

plt.figure (figsize= (6,4))
plt.rcParams.update({'font.size': 20})
bins=[0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
hist, bins, patches = plt.hist(
    np.clip(runtime_list_160, bins[0], bins[-1]),
    color='tab:orange',
    bins=bins, edgecolor='black', 
    alpha=0.7, label='sample size: 160')
hist, bins, patches = plt.hist(
    np.clip(runtime_list_20, bins[0], bins[-1]),
    color='tab:blue',
    bins=bins, edgecolor='black', 
    alpha=0.7,
    label='sample size: 20')
plt.xscale('log')
ax=plt.gca()
plt.xticks(bins[1:])
to_int=bins[1:].astype(int)
xlabels = to_int.astype(str)
ax.margins(x=0)
xlabels[-1] = ' '
xlabels[-2] += '+'
ax.set_xticklabels(xlabels)
ax.set_ylim([0,40])
ax.tick_params(axis='x', labelsize=18)
ax.tick_params(axis='y', labelsize=18)
plt.xlabel("dadi optimization runtime (CPU hours)")
plt.ylabel("Number of data sets")
plt.minorticks_off()
plt.legend(loc='upper left', frameon=False, borderpad=0)

rc('axes', linewidth=1)
plt.tick_params('both', length=5, which='major')
plt.savefig("dadi_opts_runtime.svg",bbox_inches='tight')
plt.savefig("dadi_opts_runtime.png",bbox_inches='tight')

from statistics import median, mean
print(median(runtime_list_20))
print(median(runtime_list_160))
print(mean(runtime_list_20))
print(mean(runtime_list_160))
