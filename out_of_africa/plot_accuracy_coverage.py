import re
import pickle
import glob
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from pylab import *
from sklearn.neural_network import MLPRegressor
from mapie.regression import MapieRegressor
from mapie.metrics import regression_coverage_score

# load true values
logs = [True] * 6 + [False] * 8
params = ['$ν_{Af}$', '$ν_{B}$', '$ν_{Eu0}$', '$ν_{Eu}$', '$ν_{As0}$', 
            '$ν_{As}$', '$m_{AfB}$', '$m_{AfEu}$', '$m_{AfAs}$', 
            '$m_{EuAs}$', '$T_{Af}$', '$T_{B}$', '$T_{EuAs}$', 'misid']
log_true = pickle.load(open("dadi/input_fs/true_log_params", "rb"))
true = []
for x in log_true:
    x = [10**x[i] if logs[i] else x[i] for i in range(len(x))]
    true.append(x)
    x_round = [round(num, 3) for num in x]
true = np.array(true)

# load dadi-cli results
bestfits_list = sorted(glob.glob("dadi/inference/*bestfits*"))
dadi_pred = []
for fname in bestfits_list:
    file = open(fname, "r")
    for line in file:
        if line[0] != '#':
            nums = line.split()
            x = [float(num) for num in nums][1:-1]
            dadi_pred.append(x)
            break
dadi_pred = np.array(dadi_pred)

# load donni results
trained_mlprs = []
for i in range(14):
    trained_mlprs.append(pickle.load(open(f'pipeline_output/tuned_models/param_{i+1:02d}_predictor', 'rb')))
    
# path to test file
data = pickle.load(open(f'pipeline_output/data/test_1000_theta_1000','rb'))
# subset only 30 of 1000 dict in data
test_dict = {k:data[k] for k in list(data.keys())[0:30]}
# helper method for processing test FS for prediction
def prep_fs_for_ml(input_fs):
    '''normalize and set masked entries to zeros
    input_fs: single Spectrum object from which to generate prediction'''
    # make sure the input_fs is normalized
    if round(input_fs.sum(), 3) != float(1):
        input_fs = input_fs/input_fs.sum()
    # assign zeros to masked entries of fs
    input_fs.flat[0] = 0
    input_fs.flat[-1] = 0
    return input_fs
def prep_data(data: dict, mapie=True):
    '''
    Helper method for outputing X and y from input data dict
    '''
    # require dict to be ordered (Python 3.7+)
    X_input = [np.array(fs).flatten() for fs in data.values()]
    y_label = list(data.keys())
    # parse labels into single list for each param (required for mapie)
    y_label_unpack = list(zip(*y_label)) if mapie else [y_label]
    return X_input, y_label_unpack

# process test data
prep_test_dict = {}
for params_key in test_dict:
    prep_test_dict[params_key] = prep_fs_for_ml(test_dict[params_key])
# flatten fs and parse test label
X_test, y_test = prep_data(prep_test_dict)

alpha = [0.05, 0.1, 0.2, 0.5, 0.7, 0.85]
pred_list = []
pi_list = []
for i, model in enumerate(trained_mlprs):
    pred, pis = model.predict(X_test, alpha=alpha)
    delog_pred = [10**p for p in pred] if logs[i] else pred
    delog_pis = [10**p for p in pis] if logs[i] else pis
    pred_list.append(delog_pred)
    pi_list.append(delog_pis)
donni_pred = np.array(pred_list).T

# plot accuracy
cmap_1 = cm.get_cmap('tab20')
cmap_2 = cm.get_cmap('tab20b')
hex_1 = [matplotlib.colors.rgb2hex(cmap_1(i)) for i in range(cmap_1.N)]
index_list = [0, 2, 4, 6, 9, 10, 13, 14, 17, 18]
hex_2 = [matplotlib.colors.rgb2hex(cmap_2(i)) for i in index_list]
colors = hex_1 + hex_2

def plot_accuracy_single(x, y, size=[8, 2, 20], x_label="Simulated",
                         y_label="inferred", log=False,
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
        plt.scatter(x, y, s=size[0]*2**3)  # 's' specifies dots size
    else:
        fs_list = [f'fs{i}' for i in range(1, 31)]
        for i in range(len(fs_list)):
            ax.scatter(x[i], y[i], c=c[i], label=fs_list[i], s=size[0]*2**3)
        # ax.legend(bbox_to_anchor=(1.1, 1.05), ncol=2)
        
    # axis label texts
    plt.xlabel(x_label, fontsize=size[2]+3, labelpad=size[2]/2)
    plt.ylabel(y_label, fontsize=size[2]+3, labelpad=size[2]/2)

    # only plot in log scale if log specified for the param
    if log: # for nu
        plt.xscale("log")
        plt.yscale("log")
        # axis scales customized to data
        plt.xlim([min(x+y)*10**-0.5, max(x+y)*10**0.5])
        plt.ylim([min(x+y)*10**-0.5, max(x+y)*10**0.5])
        plt.xticks(ticks=[1e-2, 1e0, 1e2])
        plt.yticks(ticks=[1e-2, 1e0, 1e2])
        plt.minorticks_off()
    else:
        # axis scales customized to data
        if 1 < max(x+y) < 4: # for T
            plt.xlim([min(x+y)-0.5, max(x+y)+0.5])
            plt.ylim([min(x+y)-0.5, max(x+y)+0.5])
            plt.xticks(ticks=[i for i in range(0, int(max(x+y))+2)])
            plt.yticks(ticks=[i for i in range(0, int(max(x+y))+2)])
        elif max(x+y) < 0.5: # for misid
            plt.xlim([min(x+y)-0.05, max(x+y)+0.05])
            plt.ylim([min(x+y)-0.05, max(x+y)+0.05])
            plt.xticks(ticks=[0,0.1,0.2,0.3])
            plt.yticks(ticks=[0,0.1,0.2,0.3])  
        else: # for m
            plt.xlim([min(x+y)-1, max(x+y)+1])
            plt.ylim([min(x+y)-1, max(x+y)+1])
            plt.xticks(ticks=[i for i in range(0, int(max(x+y)+0.5)+1, 5)])
            plt.yticks(ticks=[i for i in range(0, int(max(x+y)+0.5)+1, 5)])
    plt.tick_params('both', length=size[2]/2, which='major')
    # plot a line of slope 1 (perfect correlation)
    plt.axline((0, 0), (1, 1), linewidth=size[1]/2, color='black', zorder=-100)

    # plot scores if specified
    if rho is not None:
        plt.text(0.3, 0.78, "ρ: " + str(round(rho, 4)),
                 horizontalalignment='center', verticalalignment='center',
                 fontsize=size[2], transform=ax.transAxes)
    if title is not None:
        ax.text(0.05,0.98,title, transform=ax.transAxes, va='top', fontsize=size[2]+5)
    plt.rc('xtick', labelsize=size[2])
    plt.rc('ytick', labelsize=size[2])
    
fig1 = plt.figure(figsize=(14, 7), dpi=150)
font = {'size': 18}
plt.rc('font', **font)
# first for loop to draw dadi prediction
for i, j in zip([0,4,10,6], [0,1,2,3]):
    param_true = true[:, i]
    param_pred = dadi_pred[:, i]
    rho = stats.spearmanr(param_true, param_pred)[0]
    ax = fig1.add_subplot(2,4,j+1, aspect='equal')
    ax = plot_accuracy_single(list(param_true), list(param_pred),
                                size=[4, 1, 18], x_label="Simulated",       
                                y_label="dadi inferred", log=logs[i], 
                                rho=rho, title=f'{params[i]}',
                                c=colors)
# second for loop to draw donni prediction
for i, j in zip([0,4,10,6], [0,1,2,3]):
    param_true = true[:, i]
    param_pred = donni_pred[:, i]
    rho = stats.spearmanr(param_true, param_pred)[0]
    ax = fig1.add_subplot(2,4,j+1+4, aspect='equal')
    ax = plot_accuracy_single(list(param_true), list(param_pred),
                                size=[4, 1, 18], x_label="Simulated",       
                                y_label="donni inferred", log=logs[i], 
                                rho=rho, title=f'{params[i]}',
                                c=colors)
# plot legend

fig1.tight_layout(pad=0.5, h_pad=0)
plt.savefig("plots/accuracy.svg", transparent=True, dpi=150)
plt.savefig("plots/accuracy.png", transparent=True, dpi=150)


# plot supp figs
fig1 = plt.figure(figsize=(20, 8), dpi=150)
fig1.clear()

# first for loop to draw dadi prediction
for i, j in zip([1,2,3,5,13], [0,1,2,3,4]):
    param_true = true[:, i]
    param_pred = dadi_pred[:, i]
    rho = stats.spearmanr(param_true, param_pred)[0]
    ax = fig1.add_subplot(2,5,j+1, aspect='equal')
    
    ax = plot_accuracy_single(list(param_true), list(param_pred),
                                size=[4, 1, 18], x_label="Simulated",       
                                y_label="dadi inferred", log=logs[i], 
                                rho=rho, title=f'{params[i]}',
                                c=colors)
# first for loop to draw donni prediction
for i, j in zip([1,2,3,5,13], [0,1,2,3,4]):
    param_true = true[:, i]
    param_pred = donni_pred[:, i]
    rho = stats.spearmanr(param_true, param_pred)[0]
    ax = fig1.add_subplot(2,5,j+1+5, aspect='equal')
    ax = plot_accuracy_single(list(param_true), list(param_pred),
                                size=[4, 1, 18], x_label="Simulated",       
                                y_label="donni inferred", log=logs[i], 
                                rho=rho, title=f'{params[i]}',
                                c=colors)

fig1.tight_layout(pad=0.5, h_pad=0)
plt.savefig("plots/accuracy_supp_1.svg", transparent=True, dpi=150)
plt.savefig("plots/accuracy_supp_1.png", transparent=True, dpi=150)

fig2 = plt.figure(figsize=(20, 8), dpi=150)
fig2.clear()
# first for loop to draw dadi prediction
for i, j in zip([11,12,7,8,9], [0,1,2,3,4]):
    param_true = true[:, i]
    param_pred = dadi_pred[:, i]
    ax = fig2.add_subplot(2,5,j+1, aspect='equal')
    rho = stats.spearmanr(param_true, param_pred)[0]
    
    ax = plot_accuracy_single(list(param_true), list(param_pred),
                                size=[4, 1, 18], x_label="Simulated",       
                                y_label="dadi inferred", log=logs[i], 
                                rho=rho, title=f'{params[i]}',
                                c=colors)
# second for loop to draw donni prediction
for i, j in zip([11,12,7,8,9], [0,1,2,3,4]):
    param_true = true[:, i]
    param_pred = donni_pred[:, i]
    rho = stats.spearmanr(param_true, param_pred)[0]
    ax = fig2.add_subplot(2,5,j+1+5, aspect='equal')
    ax = plot_accuracy_single(list(param_true), list(param_pred),
                                size=[4, 1, 18], x_label="Simulated",       
                                y_label="donni inferred", log=logs[i], 
                                rho=rho, title=f'{params[i]}',
                                c=colors)

fig2.tight_layout(pad=0.5, h_pad=0)
plt.savefig("plots/accuracy_supp_2.svg", transparent=True, dpi=150)
plt.savefig("plots/accuracy_supp_2.png", transparent=True, dpi=150)


# plot coverage
all_coverage = []
true = np.array(true).T
for param in range(14):
    true_for_pi = true[param]
    pis = np.array(pi_list[param])
    coverage_scores = [
        regression_coverage_score(true_for_pi, pis[:, 0, i], pis[:, 1, i])
        for i, _ in enumerate(alpha)]
    all_coverage.append(coverage_scores)
    
def plot_coverage(cov_scores, expected, params):
    observed = []
    for cov_score in cov_scores:
        observed.append([s*100 for s in cov_score])
    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    # ax.text(0.05,0.95,f'Prediction\ninterval\ncoverage', transform=ax.transAxes, va='top', fontsize=17)
    ax.set_xlabel('Expected', fontsize=20, labelpad=10)
    ax.set_ylabel('Observed', fontsize=20)
    # set up multiple colors
    cm = plt.get_cmap('tab20')
    cm_id = reversed([0,2,3,4,6,7,8,10,11,12,14,16,18,19])
    ax.set_prop_cycle('color', [cm.colors[i] for i in cm_id])
    # plot coverage line for each param
    for i in range(len(params)):
        ax.plot(expected, observed[i], alpha=0.8,
                label=params[i],linewidth=1.25) # , marker='o', linewidth=2)
    # plot diagonal line
    ax.plot([0,100], [0,100], '-k', zorder=50, lw=1)
    # define axis range
    plt.xlim([0, 100])
    plt.ylim([0, 100])
    # define ticks
    plt.xticks(ticks=[i for i in range(0, 101, 25)])
    plt.yticks(ticks=[i for i in range(0, 101, 25)])
    plt.tick_params('both', length=10, which='major')
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    # add legend
    ax.legend(fontsize=15, frameon=False,
                labelspacing=0.25,
                handlelength=0.5,
                columnspacing=1,
                # loc='lower right',
                ncol=2,
            bbox_to_anchor=(1,1), loc="upper left")
    fig.tight_layout(pad=0.6, h_pad=0.5)

fig = plt.figure(figsize=(7, 4.5),dpi=150)
plot_coverage(all_coverage,[95, 90, 80, 50, 30, 15], params)

plt.savefig("plots/coverage.svg", transparent=True, dpi=150)
plt.savefig("plots/coverage.png", transparent=True, dpi=150)

