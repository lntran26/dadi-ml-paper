import pickle
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from mapie.metrics import regression_coverage_score

# Load test data set and trained models
theta = 1000
# path to test file
data = pickle.load(open(f'../pipeline_output/data/test_1000_theta_1000','rb'))
# subset only 100 of 1000 dict in data
test_dict = {k:data[k] for k in list(data.keys())[0:100]}

mlpr_dir_1 = "tuned_models_12369"
mlpr_dir_2 = "../pipeline_output/tuned_models"

mapie_nu = pickle.load(open(f'{mlpr_dir_2}/param_01_predictor', 'rb'))
mapie_T = pickle.load(open(f'{mlpr_dir_1}/param_02_predictor', 'rb'))

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
# normalize and change masked entries to 0
prep_test_dict = {}
for params_key in test_dict:
    prep_test_dict[params_key] = prep_fs_for_ml(test_dict[params_key])
prep_test_dict_full = {}
for params_key in data:
    prep_test_dict_full[params_key] = prep_fs_for_ml(data[params_key])

# flatten fs and parse test label
X_test, y_test = prep_data(prep_test_dict)
X_test_full, y_test_full = prep_data(prep_test_dict_full)

# separate each param; name each by the same params as above
nu_test = y_test[0]
T_test = y_test[1]
nu_test_full = y_test_full[0]
T_test_full = y_test_full[1]

# print(len(nu_test))
# print(nu_test[0])

# generate all MAPIE results
alpha = [0.05, 0.1, 0.2, 0.5, 0.7, 0.85]
# for nu
nu_pred, nu_pis = mapie_nu.predict(X_test, alpha=alpha)
# print(len(nu_pred))
# print(nu_pred[0])

nu_pred_full, nu_pis_full = mapie_nu.predict(X_test_full, alpha=alpha)
# for T
T_pred, T_pis = mapie_T.predict(X_test, alpha=alpha)
T_pred_full, T_pis_full = mapie_T.predict(X_test_full, alpha=alpha)
# for nu in log scale, exponentiate each of the test and pred values
nu_test_delog = [10**p_true for p_true in nu_test]
nu_pred_delog = [10**p_pred for p_pred in nu_pred]
# calculate rho score for nu and T
rho_nu = stats.spearmanr(nu_test, nu_pred)[0]
rho_T = stats.spearmanr(T_test, T_pred)[0]
# calculate T/nu ratio
T_over_nu = [T/nu for T,
                nu in zip(T_test, nu_test_delog)]
# calculate coverage scores from prediction intervals
nu_coverage_scores = [
    regression_coverage_score(nu_test_full, nu_pis_full[:, 0, i], nu_pis_full[:, 1, i])
    for i, _ in enumerate(alpha)]
T_coverage_scores = [
    regression_coverage_score(T_test_full, T_pis_full[:, 0, i], T_pis_full[:, 1, i])
    for i, _ in enumerate(alpha)]
# plot coverage
expected = [95, 90, 80, 50, 30, 15]
params = ['ν', 'T']
observed = ([s*100 for s in nu_coverage_scores],
            [s*100 for s in T_coverage_scores])
            
# preprocessing for plotting 95% prediction intervals (in T/nu order)
# convert pis to plot nu in original values (not log values)
nu_pis_low_delog = [10**low for low in nu_pis[:, 0, 0]]
nu_pis_high_delog = [10**high for high in nu_pis[:, 1, 0]]

nu_arr_delog = np.array([nu_test_delog, nu_pred_delog, nu_pis_low_delog, nu_pis_high_delog]) 
T_arr = np.array([T_test, T_pred, T_pis[:, 0, 0], T_pis[:, 1, 0]])

# sort by param
int_arr_all = [nu_arr_delog.T.tolist(), T_arr.T.tolist()]
size = 100
x = range(size)

# add T_over_nu column to both nu and T array and transpose to sort
nu_arr_delog_test = np.array([nu_test_delog, nu_pred_delog, nu_pis_low_delog, nu_pis_high_delog, T_over_nu]).transpose(1, 0)

T_arr_test = np.array([T_test, T_pred, T_pis[:, 0, 0], T_pis[:, 1, 0],T_over_nu]).transpose(1, 0)

# sort by the value of fifth column (T_over_nu)
nu_arr_sorted = nu_arr_delog_test[nu_arr_delog_test[:, 4].argsort()]
T_arr_sorted = T_arr_test[T_arr_test[:, 4].argsort()]

# transpose sorted array back
nu_arr_sorted = nu_arr_sorted.transpose(1, 0)
T_arr_sorted = T_arr_sorted.transpose(1, 0)

# save new T_over_nu that is sorted in the same order
T_over_nu_test = nu_arr_sorted[4].tolist()

# remove T_over_nu column after sort
int_arr_all_test = [nu_arr_sorted[:4].T.tolist(), T_arr_sorted[:4].T.tolist()]

# plot function
def plot_accuracy_single(x, y, size=[8, 2, 20], x_label="Simulated",
                         y_label="Inferred", log=False,
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
        plt.scatter(x, y, c=c, vmax=5, s=size[0]*2**3, alpha=0.8)  # vmax: colorbar limit
        cbar = plt.colorbar(fraction=0.047)
        cbar.ax.set_title(r'$\frac{T}{ν}$',
                          fontweight='bold', fontsize=size[2])

    # axis label texts
    plt.xlabel(x_label, fontsize=size[2]+3, labelpad=size[2]/2)
    plt.ylabel(y_label, fontsize=size[2]+3, labelpad=size[2]/2)

    # only plot in log scale if log specified for the param
    if log: # for nu
        plt.xscale("log")
        plt.yscale("log")
        # axis scales customized to data
        plt.xlim([min(x+y)*10**-0.2, max(x+y)*10**0.5])
        plt.ylim([min(x+y)*10**-0.2, max(x+y)*10**0.5])
        plt.xticks(ticks=[1e-2, 1e0, 1e2])
        plt.yticks(ticks=[1e-2, 1e0, 1e2])
        plt.minorticks_off()
    else: # for T
        # axis scales customized to data
        plt.xlim([min(x+y)-1, max(x+y)-0.25])
        plt.ylim([min(x+y)-1, max(x+y)-0.25])
        plt.xticks(ticks=[i for i in [0,2,4]])
        plt.yticks(ticks=[i for i in [0,2,4]])
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

# Plot
fig = plt.figure(1, figsize=(16, 12), dpi=150)
font = {'size': 25}
plt.rc('font', **font)
ax = fig.add_subplot(1,3,1, aspect='equal')
ax = plot_accuracy_single(nu_test_delog, nu_pred_delog, y_label="donni inferred", 
                            size=[6, 2, 25], log=True, rho=rho_nu, c=T_over_nu, title = 'ν')

ax = fig.add_subplot(1,3,2, aspect='equal')
ax = plot_accuracy_single(T_test, T_pred, size=[6, 2, 25], y_label="donni inferred",
                        log=False, rho=rho_T, c=T_over_nu, title = 'T')

ax = fig.add_subplot(1,3,3, aspect='equal')
ax.text(0.05,0.95,'Confidence\ninterval\ncoverage', transform=ax.transAxes, va='top', fontsize=25)
ax.set_xlabel('Expected', fontsize=28)
ax.set_ylabel('Observed', fontsize=28)

# plot coverage line for each param
for i in range(len(params)):
    ax.plot(expected, observed[i],
            label=params[i],linewidth=2) # , marker='o', linewidth=2)
# plot diagonal line
ax.plot([0,100], [0,100], '-k', zorder=-100, lw=1)
# define axis range
plt.xlim([0, 100])
plt.ylim([0, 100])
# define ticks
plt.xticks(ticks=[i for i in range(0, 101, 25)])
plt.yticks(ticks=[i for i in range(0, 101, 25)])
# plt.xticks(ticks=[i for i in [15,50,75,95]])
# plt.yticks(ticks=[i for i in [15,50,75,95]])
plt.tick_params('both', length=5, which='major')
plt.tick_params('both', length=12.5, which='major')
# add legend
ax.legend(fontsize=28, frameon=False, loc='lower right')
fig.tight_layout(pad=0.6, h_pad=0)

plt.savefig(f'plots/figures/accuracy_coverage.svg', transparent=True, dpi=150)
plt.savefig(f'plots/figures/accuracy_coverage.png', transparent=True, dpi=150)

# plot 95% prediction intervals
fontsize=12
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
logs = [True, False]
for param, int_arr, log in zip(params, int_arr_all_test, logs):
    int_arr = np.array(int_arr[:size]) # size=100
    int_arr = int_arr.transpose(1, 0)
    fig = plt.figure(figsize=(8, 2))
    ax = plt.gca()
    ax.margins(x=0.01)
    plt.rc('font', size=fontsize)
    
    # color the range of true parameter value
    ax.fill_between(x, min(int_arr[0]), max(int_arr[0]), alpha=0.2)
    # only plot in log scale if log
    if log:
        plt.yscale("log")

    ax.tick_params('y', length=4, which='major', labelsize=fontsize)
    ax.get_xaxis().set_visible(False) # whether to have x axis
    plt.minorticks_off()
    
    # x axis title
    ax.text(0.2,0.1,f'95% confidence intervals for 100 test AFS', transform=ax.transAxes, va='top', fontsize=fontsize)

    # color bar formatting
    cbaxes = fig.add_axes([0.2, 0.35, 0.1, 0.02]) 
    # new axes dimensions [(right from) left, (up from) bottom, width, height]
    # plot true values
    cbar = fig.colorbar(ax.scatter(x, int_arr[0], s=2, c=T_over_nu_test, vmax=5,zorder=2.5), cax=cbaxes, orientation='horizontal', ticks=[0.1,5])
    cbar.set_label('Simulated, by T/ν', fontsize=fontsize, loc='center',labelpad=1)
    cbar.ax.tick_params(labelsize=fontsize) # font size for cbar tick label

    # plot prediction values and interval bars
    neg_int = int_arr[1] - int_arr[2]
    pos_int = int_arr[3] - int_arr[1]
    ax.errorbar(x, int_arr[1], 
                yerr=[
                abs(neg_int), abs(pos_int)], 
                # fmt='o',
                fmt='.',
                markersize='2',
                elinewidth=0.5, 
                label = 'inferred value, with interval', c="tab:brown")
    ax.text(0.03,0.98,param, transform=ax.transAxes, va='top', fontsize=fontsize)
    plt.savefig(f'plots/figures/95PI_{param}_new.svg', transparent=True, dpi=150)
    plt.savefig(f'plots/figures/95PI_{param}_new.png', transparent=True, dpi=150)
    
