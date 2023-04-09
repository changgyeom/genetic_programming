import pandas as pd
import numpy as np
import datetime
import random
import copy
from tree_algorithms import *
import multiprocessing
import multiprocessing.pool
import pickle
import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

def data_prepocessing(prc_df, start_date, end_date):
    data = prc_df.loc[prc_df['Date'] <= end_date, :]
    data = data.loc[data['Date'] >= start_date, :]
    
    outlier1s = []
    outlier2s = []
    data.loc[:,'C/pC1'] = data['Close']/data['pC1']
    for date, cpc1, nc1c in zip(data['Date'], data['C/pC1'], data['nextC1/C']):
        o1 = abs(cpc1-1)
        o2 = abs(nc1c-1)
        if date < 20150615:
            outlier1 = o1 > 0.145
        else:
            outlier1 = o1 > 0.29
            
        if date < 20150614:
            outlier2 = o2 > 0.15
        else:
            outlier2 = o2 > 0.30
        outlier1s.append(outlier1)
        outlier2s.append(outlier2)
    
    data.loc[:,'outlier1'] = outlier1s
    data.loc[:, 'outlier2'] = outlier2s
    data.loc[:, 'outlier'] = np.logical_or(data['outlier1'], data['outlier2'])
    data = data.loc[data['outlier'] == False, :]
    data.drop(columns=['outlier', 'C/pC1', 'outlier1', 'outlier2'], inplace=True)
    data = data.dropna()
    return data

def tournament_selection(algorithms, k, N):
    parents_i = [random.randint(0, N-1) for _ in range(k)]
    max_i, max_fit = -1, -999999999
    for i in parents_i:
        if algorithms[i][1] > max_fit:
            max_i = i
            max_fit = algorithms[i][1]            
    return max_i

def replacement(algorithms, elem, N):
    min_i, min_fit = -1, 999999999
    for i in range(N):
        if algorithms[i][1] < min_fit:
            min_i = i
            min_fit = algorithms[i][1]
    
    if min_fit < elem[1]:
        algorithms[min_i] = elem

    return algorithms

def make_fitness(a, data, start_date, calc_num_processess):
    day_returns, day_signals = a.fitness(data, start_date, calc_num_processes)
    fit = np.sum(day_returns)
    print( 'Ended', a, fit, ' process')
    
    return (a, fit)

def init_tree_algorithms(data, start_date, filename, num_processes, calc_num_processes,  N, init_max_height, init_min_height, init_generate_ratio):
    algos = []
    global make_fitness
    print('----- generate algorithms -----')
    while(len(algos) < N):
        a = Algorithm(init_max_height, init_min_height, init_generate_ratio)
        a.algorithm_init()
        algos.append([a, data, start_date, calc_num_processes])

    print('----- make fitnesses -----')
    s = datetime.datetime.now()
    pool = MyPool(num_processes)
    print('----- make pool -----')
    init_algorithms = pool.starmap(make_fitness, algos)
    print('init ended')
    pool.close()
    pool.join()
    
    time = (datetime.datetime.now() - s).seconds
    print('----- ended : {} min -----'.format(time/60))
    with open('./save_algorithms/{}'.format(filename), 'wb') as f:
        pickle.dump(init_algorithms, f)

    #if len(init_algorithms) % 100 == 0:
    #    print(len(init_algorithms))
    return init_algorithms

def train(algorithms, data, start_date, filename, generation, N, k, max_height, mutation_rate, calc_num_processes):
    
    mean_fits = []
    min_fits = []
    max_fits = []
    
    # search
    for g in range(generation): # terminate condition
        # selection
        p1_i = tournament_selection(algorithms, k, N)
        p2_i = tournament_selection(algorithms, k, N)

        elem = copy.deepcopy(algorithms[p1_i])

        # crossover
        try_crossover = 0
        crossover = False

        while(crossover == False and try_crossover < 5):
            a1 = copy.deepcopy(algorithms[p1_i][0])
            a2 = copy.deepcopy(algorithms[p2_i][0])
            a1.xover(a2)

            day_returns1, day_signals1 = a1.fitness(data, start_date, calc_num_processes)
            fit1 = np.sum(day_returns1)

            day_returns2, day_signals2 = a2.fitness(data, start_date, calc_num_processes)
            fit2 = np.sum(day_returns2)

            elem = (a1, fit1)
            if fit1 < fit2:
                elem = (a2, fit2)

            if (elem[0].tree_height() <= max_height) and not np.isnan(elem[1]):
                crossover = True

            try_crossover+=1

        if try_crossover == 5:
            print(g, 'failed xover')
        else:
            print(g, 'successed xover')

        # mutation
        if elem[0].mutation(mutation_rate) > 0:
            #print('in')
            day_returns, day_signals = elem[0].fitness(data, start_date, calc_num_processes)
            fit = np.sum(day_returns)
            elem = (elem[0], fit)

        # replacement
        replacement(algorithms, elem, N)

        # evaluate
        fits = [algorithms[i][1] for i in range(N)]
        fit_mean = np.sum(fits)/N
        fit_max = np.max(fits)
        fit_min = np.min(fits)

        mean_fits.append(fit_mean)
        max_fits.append(fit_max)
        min_fits.append(fit_min)


        if g%10 == 0:
            print(g, fit_mean, fit_max, fit_min)
        if g%100 == 0:
            with open('./save_fitness/{}_{}.pkl'.format(filename[:-4], g), 'wb') as f:
                pickle.dump([mean_fits, max_fits, min_fits], f)
            with open('./save_algorithms/{}_{}.pkl'.format(filename[:-4], g), 'wb') as f:
                pickle.dump(algorithms, f)
                
    with open('./save_fitness/{}_{}.pkl'.format(filename[:-4], g), 'wb') as f:
        pickle.dump([mean_fits, max_fits, min_fits], f)
    with open('./save_algorithms/{}_{}.pkl'.format(filename[:-4], g), 'wb') as f:
        pickle.dump(algorithms, f)
        
    return algorithms

        
if __name__ == "__main__":    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--init', help='proceed init algorithms', required=True, type=str2bool) 
    parser.add_argument('--start_date', help='train start date', required=True, type=int) 
    parser.add_argument('--end_date', help='train end date', required=True, type=int) 
    parser.add_argument('--N', help='num of population', required=True, type=int) 
    parser.add_argument('--k', help='tournament selection', required=True, type=int) 
    
    parser.add_argument('--init_max_h', help='init tree height', required=True, type=int) 
    parser.add_argument('--train_max_h', help='train tree height', required=True, type=int) 
    
    parser.add_argument('--num_processes', help='core num multi processing', required=True, type=int) 
    parser.add_argument('--sub_num_processes', help='core sub num multi processing', required=True, type=int) 
    parser.add_argument('--calc_num_processes', help='core calc num multi processing', required=True, type=int) 
    parser.add_argument('--exp_num', help='experiment number', required=True, type=str) 

    args = parser.parse_args()
    
    # parameter settings
    N = args.N 
    k = args.k 
    init_max_height = args.init_max_h
    init_min_height = 1
    init_generate_ratio = 0.5

    start_date = args.start_date
    prev_period = 20000
    end_date = args.end_date
    time = 0
    exp_type = args.exp_num 
    num_processes = args.num_processes
    sub_num_processes = args.sub_num_processes
    calc_num_processes = args.calc_num_processes
    generation = 10000
    max_height = args.train_max_h
    mutation_rate = 0.01
    
    init_filename = exp_type + 'init_algorithms_{}_{}_{}_{}.pkl'.format(N, k, start_date, end_date)
    train_filename = exp_type + 'algorithms_{}_{}_{}_{}.pkl'.format(N, k, start_date, end_date)
    
    # data loading and preprocessing
    prc_df = pd.read_csv('../prc_jam_data_20050000_20110000.csv', index_col=None, encoding='euc-kr', dtype = {'Code_' : str})
    print('readed data')
    data = data_prepocessing(prc_df, start_date-prev_period, end_date)
    print('processed data')
    
    # make init algorithms
    print('init :',args.init)
    if args.init:
        print('init algorithms start / num processes : {} / calc num processes : {}'.format(num_processes, sub_num_processes))
        init_algorithms = init_tree_algorithms(data, start_date, init_filename, num_processes, sub_num_processes, N, init_max_height, init_min_height, init_generate_ratio)

        # rebuild algorithms for wrong algorithms
        print('check algorithms')
        for a in init_algorithms:
            if np.isnan(a[1]):
                init_algorithms.remove(a)
                print(a)

        print('add init algorithms')    
        while(len(init_algorithms) < N):
            a = Algorithm(init_max_height, init_min_height, init_generate_ratio)
            a.algorithm_init()
            
            s = datetime.datetime.now()
            day_returns, day_signals = a.fitness(data, start_date, calc_num_processes)
            fit = np.sum(day_returns)
            time += (datetime.datetime.now() - s).seconds

            if not np.isnan(fit):
                init_algorithms.append((a, fit))
        
        # save init trees
        with open('./save_algorithms/{}'.format(init_filename), 'wb') as f:
            pickle.dump(init_algorithms, f)

    else:
        # using saved trees
        with open('./save_algorithms/'+init_filename, 'rb') as f:
            init_algorithms=pickle.load(f)
        
    # train with init algorithms
    print('train start / calc num processes : {}'.format(calc_num_processes))
    algorithms = copy.deepcopy(init_algorithms)
    train(algorithms, data, start_date, train_filename, generation, N, k, max_height, mutation_rate, calc_num_processes)
    
    
    
    
    
    
