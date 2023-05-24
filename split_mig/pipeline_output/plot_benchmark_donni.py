from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def get_time_diff(str1, str2, n_cpus):
    time_1 = datetime.strptime(str1.strip(), '%a %b %d %H:%M:%S %Z %Y')
    time_2 = datetime.strptime(str2.strip(), '%a %b %d %H:%M:%S %Z %Y')
    time_diff = time_2 - time_1
    time_diff_in_cpus_h = time_diff.total_seconds()/60.0/60.0*n_cpus
    return time_diff_in_cpus_h

# data processing
generate_data = []
tuning = []
training = []

##### ns20
# generating data/train_5000
g1="Sun Feb 12 11:43:36 MST 2023"
g2="Sun Feb 12 12:10:56 MST 2023"
generate_data.append(get_time_diff(g1, g2, 10))
# tuning
tu1="Sun Feb 12 13:34:05 MST 2023"
tu2="Sun Feb 12 13:43:32 MST 2023"
tuning.append(get_time_diff(tu1, tu2, 10))
# training
tr1="Sun Feb 12 14:25:39 MST 2023"
tr2="Sun Feb 12 14:26:29 MST 2023"
training.append(get_time_diff(tr1, tr2, 10))

##### ns160
# generating data/train_5000
g3="Sat Feb 11 22:08:45 MST 2023"
g4="Sun Feb 12 03:35:29 MST 2023"
generate_data.append(get_time_diff(g3, g4, 10))
# tuning
tu3="Sun Feb 12 03:35:29 MST 2023"
tu4="Sun Feb 12 11:43:15 MST 2023"
tuning.append(get_time_diff(tu3, tu4, 10))
#training
tr3="Sun Feb 12 11:43:15 MST 2023"
tr4="Sun Feb 12 12:09:21 MST 2023"
training.append(get_time_diff(tr3, tr4, 10))

# plot
groups = ['sample size:\n20         ','sample size:\n160        ']
mlpr_runtime = pd.DataFrame({'Groups': groups, 'Generate data': generate_data, 
                      'Tuning': tuning, 'Training': training})
plt.figure()
plt.rcParams.update({'font.size': 20})
ax = mlpr_runtime.plot(kind='barh', stacked=True, edgecolor='black',alpha=0.8)
y_pos = np.arange(len(groups))
ax.set_yticks(y_pos, labels=groups)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('donni pipeline runtime (CPU hours)')
ax.set_xlim(0, 180)
plt.tick_params('x', length=10, which='major')
plt.tick_params('y', left=False)
plt.legend(loc='upper right',frameon=False, borderpad=0)

# Add labels to each bar
totals = mlpr_runtime.sum(axis=1) # Sum up the rows of our data
x_offset = 10 # bump the label up a bit above the bar.
plt.text(totals[0] + x_offset+5, totals.index[0], round(totals[0],2), ha='center')
plt.text(totals[1] + x_offset*2, totals.index[1], round(totals[1],2), ha='center')
plt.tight_layout()
plt.savefig("donni_runtime.svg", bbox_inches='tight')
plt.savefig("donni_runtime.png", bbox_inches='tight')

