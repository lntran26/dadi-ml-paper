import pickle
import numpy as np
from scipy import stats
from mapie.metrics import regression_coverage_score
from matplotlib import pyplot as plt

# Load trained models
mlpr_dir = '/xdisk/rgutenk/lnt/projects/donni_paper/split_mig/pipeline_output/ns_20/tuned_models_1'
trained_mlprs = []
for i in range(4):
    trained_mlprs.append(pickle.load(open(f'{mlpr_dir}/param_{i+1:02d}_predictor','rb')))
    
# Load msprime test data set
seq_l = 1e8
# recomb_list = [1e-8, 1e-9, 1e-10, 1e-11]
recomb_list = [1e-8, 1e-9, 1e-10]
test_data = []
for recomb in recomb_list:
    test_d = pickle.load(
        open(f'data/{seq_l:.1e}_{recomb:.1e}', 'rb'))
    test_data.append(test_d)
    
# process true values
log_true = list(test_data[0].keys())
logs = [True] * 2 + [False] * 2
true = []
for x in log_true:
    x = [10**x[i] if logs[i] else x[i] for i in range(len(x)-1)] # remove misid
    true.append(x)
true = np.array(true)

# process test FS for prediction
all_test_fs = []
for data in test_data:
    # unpack test data set
    X_test = [data[params].data.flatten()
            for params in data]
    all_test_fs.append(X_test)
    
# implementing MAPIE test
all_preds = []
all_pis = []
alpha = [0.05, 0.1, 0.2, 0.5, 0.7, 0.85]
for test in all_test_fs: # 3 different recomb
    preds, pis = [], []
    for mlpr, log in zip(trained_mlprs, logs): # 4 demo param
        pred, pi = mlpr.predict(test, alpha=alpha) # 100 test
        delog_pred = [10**p for p in pred] if log else pred
        preds.append(delog_pred)
        pis.append(pi)
    # TO-DO: add PIs info and delog PIs
    all_preds.append(np.array(preds).T)
    all_pis.append(pis)

# mapie coverage
all_coverages = []
for recomb in all_pis: # recomb is (4, 100, 2, 6)
    all_coverage = []
    for param in range(4):
        true_for_pi = np.array(log_true).T[param]
        pis = recomb[param]
        coverage_scores = [
            regression_coverage_score(true_for_pi, pis[:, 0, i], pis[:, 1, i])
            for i, _ in enumerate(alpha)]
        all_coverage.append(coverage_scores)
    all_coverages.append(all_coverage)


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
        plt.scatter(x, y, c=c, s=size[0]*2**3, cmap='tab10', alpha=0.8)

    # axis label texts
    plt.xlabel(x_label, fontsize=size[2]+3, labelpad=size[2]/2)
    plt.ylabel(y_label, fontsize=size[2]+3, labelpad=size[2]/2)

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
    else:
        # axis scales customized to data
        if max(x+y) < 4: # T
            plt.xlim([min(x+y)-0.5, max(x+y)+0.5])
            plt.ylim([min(x+y)-0.5, max(x+y)+0.5])
            plt.xticks(ticks=[i for i in range(0, int(max(x+y)+1.5))])
            plt.yticks(ticks=[i for i in range(0, int(max(x+y)+1.5))])
        else: # m
            plt.xlim([min(x+y)-1, max(x+y)+1])
            plt.ylim([min(x+y)-1, max(x+y)+1])
            plt.xticks(ticks=[i for i in range(0, int(max(x+y)+1.5), 5)])
            plt.yticks(ticks=[i for i in range(0, int(max(x+y)+1.5), 5)])
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
    
def plot_coverage(cov_scores, expected, recomb, params):
    observed = []
    for cov_score in cov_scores:
        observed.append([s*100 for s in cov_score])
    # fig = plt.figure(1, figsize=(16, 12), dpi=150)
    # ax = fig.add_subplot(1,3,3, aspect='equal')
    # plt.figure()
    ax = plt.gca()
    ax.set_aspect('equal', 'box')
    # ax.text(0.05,0.95,f'Prediction\ninterval\ncoverage\nr={recomb}', transform=ax.transAxes, va='top', fontsize=17)
    ax.set_xlabel('Expected', fontsize=22, labelpad=10)
    ax.set_ylabel('Observed', fontsize=22)
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
    plt.tick_params('both', length=10, which='major')
    plt.rc('xtick', labelsize=22)
    plt.rc('ytick', labelsize=22)
    # add legend
    ax.legend(fontsize=15, frameon=False, loc='lower right')
    fig.tight_layout(pad=0.6, h_pad=0)
    
params = ['$ν_1$', '$ν_2$','T','m']
expected = [95, 90, 80, 50, 30, 15]
recomb_list = [1e-8, 1e-9, 1e-10]

fig = plt.figure(1, figsize=(17, 12), dpi=150)
font = {'size': 20}
plt.rc('font', **font)

# first for loop to draw MLPR prediction for r=1e-8
for i, recomb in enumerate(recomb_list):
    for j, param in enumerate(params):
        if j==0:
            ax = fig.add_subplot(3,4,i*4+1, aspect='equal')
            ax = plot_coverage(all_coverages[i], expected, recomb_list[i], params)
        elif j==1:
            param_true = true[:, 0]
            param_pred = all_preds[i][:, 0]
            rho = stats.spearmanr(param_true, param_pred)[0]
            ax = fig.add_subplot(3,4,i*4+j+1, aspect='equal')
            ax = plot_accuracy_single(list(param_true), list(param_pred),
                                        size=[5, 1, 20],
                                        x_label="Simulated",
                                        y_label="donni inferred", 
                                        log=logs[0], 
                                        rho=rho, title=f'{params[0]}')
        else:
            param_true = true[:, j]
            param_pred = all_preds[i][:, j]
            rho = stats.spearmanr(param_true, param_pred)[0]
            ax = fig.add_subplot(3,4,i*4+j+1, aspect='equal')
            ax = plot_accuracy_single(list(param_true), list(param_pred),
                                        size=[5, 1, 20],
                                        x_label="Simulated",      
                                        y_label="donni inferred", 
                                        log=logs[j], 
                                        rho=rho, title=f'{param}')

fig.tight_layout(pad=0.6, h_pad=0)
plt.savefig("plots/split_mig_accuracy_coverage_msprime.pdf", transparent=True, dpi=150)
plt.savefig("plots/split_mig_accuracy_coverage_msprime.svg", transparent=True, dpi=150)
