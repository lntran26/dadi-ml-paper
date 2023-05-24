import time
import pickle
from multiprocessing import Pool
import dadi
import msprime


def msprime_split_mig(s1, p):
    (nu1, nu2, T, m, _) = p # ignore misid value
    dem = msprime.Demography()
    dem.add_population(name="A", initial_size=s1*10**nu1)  # pop1 at present time
    dem.add_population(name="B", initial_size=s1*10**nu2)  # pop2 at present time
    dem.add_population(name="C", initial_size=s1)  # ancestral pop
    dem.add_population_split(time=2*s1*T, derived=["A", "B"], ancestral="C")
    dem.set_symmetric_migration_rate(["A", "B"], m/(2*s1))
    return dem


def msprime_generate_ts(args):
    '''Simulate TS under msprime demography model'''
    (dem, ns, ploidy, seq_l, recomb) = args
    # simuate tree sequences
    return(msprime.sim_ancestry(samples=ns, ploidy=ploidy, demography=dem,
                                sequence_length=seq_l,
                                recombination_rate=recomb))


def msprime_generate_data(params_list, dem_list, ns, ploidy, seq_l,
                          recomb, mut, sample_nodes=None, ncpu=None):
    '''Parallelized version for generating data from msprime 
    using multiprocessing.
    Output format same as generate_data with dadi but FS were simulated
    and summarized from TS data generated under msprime models.'''
    arg_list = [(dem, ns, ploidy, seq_l, recomb) for dem in dem_list]
    with Pool(processes=ncpu) as pool:
        ts_list = pool.map(msprime_generate_ts, arg_list)

    data_dict = {}
    for params, ts in zip(params_list, ts_list):
        # simulate mutation to add variation
        mts = msprime.sim_mutations(ts, rate=mut, discrete_genome=False)
        # Using discrete_genome=False means that the mutation model will
        # conform to the classic infinite sites assumption,
        # where each mutation in the simulation occurs at a new site.

        # convert mts to afs
        afs = mts.allele_frequency_spectrum(sample_sets=sample_nodes,
                                            polarised=True, span_normalise=False)
        # polarised=True: generate unfolded/ancestral state known fs
        # span_normalise=False: by default, windowed statistics are divided by the
        # sequence length, so they are comparable between windows.
        
        # convert afs to dadi fs object, normalize and save
        fs = dadi.Spectrum(afs)
        if fs.sum() == 0:
            pass
        else:
            data_dict[params] = fs/fs.sum()
    return data_dict


# msprime demographic and ancestry simulation parameters (unchanged params)
s1 = 1e4 # ancestral pop size
ns = {"A": 10, "B":10} # sample size
ploidy = 2 # diploid
mut = 1e-8 # mutation rate

# load test params (nu1, nu2, T, m, misid) from comparable test set
test_params = pickle.load(open(f'/xdisk/rgutenk/lnt/projects/donni_paper/split_mig/dadi/ns_20/input_fs/true_log_params', 'rb'))

# make dem_list for msprime from list of test_params
dem_list = [msprime_split_mig(s1, p) for p in test_params]

# other required msprime variables
seq_l = 1e8
recomb_list = [1e-8, 1e-9, 1e-10]

# need to make list of sample node ids to convert from ts to joint afs
s0 = [node_id for node_id in range(0,20)]
s1 = [node_id for node_id in range(20,40)]
sample_nodes = [s0, s1]

# generate test data dictionary
for recomb in recomb_list:
    print(f'Sequence length: {seq_l:.1e}; Recombination rate: {recomb:.1e}')
    start = time.time()
    test_d = msprime_generate_data(test_params, dem_list, ns, ploidy, seq_l, 
                                    recomb, mut, sample_nodes)
    print('msprime execution time: {0:.2f}s\n'.format(time.time()-start))
    pickle.dump(test_d, open(f'data/{seq_l:.1e}_{recomb:.1e}', 'wb'))
    