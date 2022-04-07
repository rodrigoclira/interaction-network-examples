import numpy as np
from collections import defaultdict
import pandas as pd
from os import chdir, getcwd
from copy import deepcopy
from scipy.stats import entropy


def parse_files(simu_size, simu_path_pattern, best_fitness_path=None, aslist=False):
    if aslist:
        results = []
    else:
        results = defaultdict(dict)

    best_fitness_df = None
    if best_fitness_path:
        best_fitness_df = pd.read_csv(best_fitness_path)

    for simulation in range(0,simu_size):
        #filename = folder+results_file%simulation
        filename = simu_path_pattern%simulation
        data = np.loadtxt(filename)
        agents = len(data[0])
        data_transformed = data.reshape(agents, agents)
        dict_data = {'data': data_transformed,
                    'data_normalized': data_transformed / np.sqrt((np.sum(data_transformed**2))),
                    'simulation': simulation,
                    }

        if best_fitness_df:
            fit = best_fitness_df['simulations'][simulation]
            dict_data['fit'] = fit

        if aslist:
            results.append(dict_data)
        else:
            results[simulation] = dict_data

    return results

def sort_results(results, key="fit"):
    if type(results) != type([]):
        raise TypeError()

    result_ordered = sorted(results, key = lambda result: result[key])
    return result_ordered

def calculate_tw(data, tws, skips, max_iter, accumulated=True, debug = False):
    results = defaultdict(dict)
    for tw in tws:
        twi=0
        twf=tw
        for skip in skips:
            results[tw][skip] = []
            if debug:
                print (f"\nCalculating TW to tw={tw} skip={skip}")
            while twf <= max_iter:#max_iter-tw + 1:
                
                if accumulated:
                    start = 0
                else:                    
                    start = twi
                 
                in_tw = sum(data[start:twf])
                
                if debug:
                    print(f"{start}:{twf}", end="  ")

                results[tw][skip].append(in_tw)
                twi+=skip
                twf=twi+tw
    if debug:
        for tw in tws:
            for skip in skips:
                print(f'| t{tw} s{skip} len ->{len(results[tw][skip])}')

    return results

def worst_best_simulation(results_path, verbose = True, sep=','):
    if verbose:
        print(f"-> Loading {results_path}...")
    best_fitness_df = pd.read_csv(results_path, sep=sep)
    worst_fit_id = best_fitness_df['simulations'].idxmax()
    best_fit_id= best_fitness_df['simulations'].idxmin()
    return worst_fit_id, best_fit_id

def load_file_tw(file_path, normalized = True, debug=False):
    if debug:
        print(f"-> Loading {file_path}...")
    data = np.loadtxt(file_path)
    iterations = len(data)
    agents = int(np.sqrt(len(data[0])))
    data_simulation = data.reshape(iterations, agents, agents)
    if normalized: 
        #data_simulation = data_simulation / np.sqrt((np.sum(data_simulation**2)))
        cont = 0 
        while cont < iterations:
            if data_simulation[cont].sum()!=0:
                data_simulation[cont] =  data_simulation[cont] / np.sqrt((np.sum(data_simulation[cont]**2)))
            cont+=1
    #print(iterations)
    return data_simulation


def calculate_metric_tw(data_tw, tws, skips, metric, in_colab):
    if in_colab:
        from portrait_divergence import portrait_divergence
    else:
        from portraitdivergence.portrait_divergence import portrait_divergence

    result_div = defaultdict(dict)
    for tw in tws:
        for skip in skips:
            cont_b = 0
            size = len(data_tw[tw][skip])
            result_div[tw][skip] = np.zeros((size, size))
            while cont_b < size:
                cont_w = 0
                while cont_w < size:
                    tw_simu_b = data_tw[tw][skip][cont_b]
                    tw_simu_w = data_tw[tw][skip][cont_w]
                    if metric == 'pd':
                        result_div[tw][skip][cont_b][cont_w] = portrait_divergence(tw_simu_b, tw_simu_w)
                    elif metric == 'kd':
                        result_div[tw][skip][cont_b][cont_w] = entropy(np.hstack(tw_simu_b), np.hstack(tw_simu_w))
                    #gc.collect()
                    cont_w+=1
                cont_b+=1
    return result_div


def readLastLine(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
        last_line = lines[-1]
    #print(last_line)
    return last_line


if __name__ == '__main__':
    print(getcwd())
    data = load_file_tw("analysis_interaction_graph/gwo_results/GWO_Griewank_intype_euclidian_improved_False_init_Uniform_it_500_dim_50_swarm_20_eval_10000_sim_07__interaction_network_by_iteration.int_net")
    tws = [25, 100, 200]
    skips = [1]
    max_iter = 1500
    data_tw = calculate_tw(data,tws, skips, max_iter)
    for tw in tws:
        for skip in skips:
            print(f"\n{tw} {skip} len={len(data_tw[tw][skip])}")
