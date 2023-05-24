import pickle
import glob
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score

# load true params from file
log_true = pickle.load(open("dadi/ns_20/input_fs/true_log_params", "rb"))
logs = [True] * 2 + [False] * 3
true = []
for x in log_true:
    x = [10**x[i] if logs[i] else x[i] for i in range(len(x))]
    true.append(x)
true = np.array(true)

# load dadi inference
bestfits_list = sorted(glob.glob("dadi/ns_20/inference/*bestfits*"))
dadi_pred = []
for fname in bestfits_list:
    file = open(fname, "r")
    for line in file:
        if line[0] != '#':
            nums = line.split()
            x = [float(num) for num in nums][1:-1]
            # remove ll (first) and theta (last)
            dadi_pred.append(x)
            break
dadi_pred = np.array(dadi_pred)

# Load trained models
mlpr_dir = 'pipeline_output/ns_20/tuned_models_1'
trained_mlprs = []
for i in range(5):
    trained_mlprs.append(pickle.load(open(f'{mlpr_dir}/param_{i+1:02d}_predictor','rb')))
# load test data
data = pickle.load(
        open(f'pipeline_output/ns_20/data/test_100_theta_1000', 'rb'))
# process test FS for prediction
X_test = [data[params].data.flatten() for params in data]
# get prediction
mlpr_pred = []
for mlpr, log in zip(trained_mlprs, logs):
    pred = mlpr.predict(X_test)
    delog_pred = [10**p for p in pred] if log else pred
    mlpr_pred.append(delog_pred)
mlpr_pred = np.array(mlpr_pred).T
    
def plot_accuracy_single(x, y, size=[8, 2, 20], x_label="true",
                         y_label="predict", log=False,
                         r2=None, msle=None, rho=None, c=None, title=None):
    '''
    Plot a single x vs. y scatter plot panel, with correlation scores

    x, y = lists of x and y values to be plotted, e.g. true, pred
    size = [dots_size, line_width, font_size],
        e.g size = [8,2,20] for 4x4, size= [20,4,40] for 2x2
    log: if true will plot in log scale
    r2: r2 score for x and y
    msle: msle score for x and y (x, y need to be non-log, i.e. non-neg)
    rho: rho score for x and y
    c: if true will plot data points in a color range with color bar
    '''

    ax = plt.gca()
    # make square plots with two axes the same size
    ax.set_aspect('equal', 'box')

    # plot data points in a scatter plot
    if c is None:
        plt.scatter(x, y, s=size[0]*2**3, alpha=0.8)  # 's' specifies dots size
    else:  # condition to add color bar
        # this is for the converged vs not converged settings
        plt.scatter(x, y, s=size[0]*2**3, c=c, cmap='tab10', alpha=0.8)

    # axis label texts
    plt.xlabel(x_label, fontsize=size[2], labelpad=size[2]/2)
    plt.ylabel(y_label, fontsize=size[2], labelpad=size[2]/2)

    # only plot in log scale if log specified for the param
    if log:
        plt.xscale("log")
        plt.yscale("log")
        # axis scales customized to data
        plt.xlim([min(x+y)*10**-0.5, max(x+y)*10**0.5])
        plt.ylim([min(x+y)*10**-0.5, max(x+y)*10**0.5])
        plt.xticks(ticks=[1e-2, 1e0, 1e2])
        plt.yticks(ticks=[1e-2, 1e0, 1e2])
        plt.minorticks_off()
    elif max(x+y) > 5: # m
        plt.xlim([min(x+y)-1, max(x+y)+1])
        plt.ylim([min(x+y)-1, max(x+y)+1])
        plt.xticks(ticks=[i for i in range(0, int(max(x+y)+1.5), 5)])
        plt.yticks(ticks=[i for i in range(0, int(max(x+y)+1.5), 5)])
    elif 1 < max(x+y) < 4: # T
        plt.xlim([min(x+y)-0.5, max(x+y)+0.5])
        plt.ylim([min(x+y)-0.5, max(x+y)+0.5])
        plt.xticks(ticks=[i for i in range(0, int(max(x+y))+2)])
        plt.yticks(ticks=[i for i in range(0, int(max(x+y))+2)])
    else: # misid
        plt.xlim([min(x+y)-0.05, max(x+y)+0.05])
        plt.ylim([min(x+y)-0.05, max(x+y)+0.05])
        plt.xticks(ticks=[0, 0.1, 0.2, 0.3])
        plt.yticks(ticks=[0, 0.1, 0.2, 0.3])

    plt.tick_params('both', length=size[2]/2, which='major')
    # plot a line of slope 1 (perfect correlation)
    plt.axline((0, 0), (1, 1), linewidth=size[1]/2, color='black', zorder=-100)

    # plot scores if specified
    if r2 is not None:
        plt.text(0.7, 0.3, "\n\n" + r'$R^{2}$: ' + str(round(r2, 4)),
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=size[2], transform=ax.transAxes)
    if rho is not None:
        plt.text(0.3, 0.78, "ρ: " + str(round(rho, 4)),
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=size[2], transform=ax.transAxes)
    if title is not None:
        ax.text(0.05,0.98,title, transform=ax.transAxes, va='top', fontsize=size[2])
    plt.rc('xtick', labelsize=size[2])
    plt.rc('ytick', labelsize=size[2])
    # thickness of ticks
    ax.xaxis.set_tick_params(width=1)
    ax.yaxis.set_tick_params(width=1)   
    
params = ['$ν_1$', '$ν_2$', 'T', 'm', 'misid']
fig = plt.figure(1, figsize=(10, 4), dpi=150)
from pylab import *
rc('axes', linewidth=1) # line width for sub fig box

# first for loop to draw MLPR prediction
for i, param in enumerate(params):
    param_true = true[:, i]
    param_pred = dadi_pred[:, i]
    rho = stats.spearmanr(param_true, param_pred)[0]
    ax = fig.add_subplot(2,5,i+1, aspect='equal')
    ax = plot_accuracy_single(list(param_true), list(param_pred),
                                size=[1.5, 1, 10], x_label="Simulated",       
                                y_label="dadi inferred ", log=logs[i], 
                                rho=rho, title=f'{param}')

# second for loop to draw dadi prediction

for i, param in enumerate(params):
    param_true = true[:, i]
    param_pred = mlpr_pred[:, i]
    rho = stats.spearmanr(param_true, param_pred)[0]
    ax = fig.add_subplot(2,5,i+1+5, aspect='equal')
    ax = plot_accuracy_single(list(param_true), list(param_pred),
                                size=[1.5, 1, 10], x_label="Simulated",       
                                y_label="donni inferred", log=logs[i],
                                rho=rho, title=f'{param}')

fig.tight_layout()
plt.savefig("split_mig_accuracy.svg", bbox_inches='tight', dpi=150)
plt.savefig("split_mig_accuracy.png", bbox_inches='tight', dpi=150)

