import pandas as pd
import numpy as np
import random
import gc
import multiprocessing
    
class node(object):
    
    def __init__(self, value, left=None, right=None, is_leaf=False, upper_comp=False):
        self.value = value
        self.value_type = value.inst_type
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.upper_comp = upper_comp
        
class inst1(object):
    num_childs = 2
    inst_type = "function"
    def __call__(self, *arg):
        return arg[0] + arg[1]
    def __repr__(self):
        return "+"
    
class inst2(object):
    num_childs = 2
    inst_type = "function"
    def __call__(self, *arg):
        return arg[0] - arg[1]
    def __repr__(self):
        return "-"

class inst3(object):
    num_childs = 2
    inst_type = "function"
    def __call__(self, *arg):
        return arg[0] * arg[1]
    def __repr__(self):
        return "*"

class inst4(object):
    num_childs = 2
    inst_type = "function"
    def __call__(self, *arg):
        return arg[0] / arg[1]
    def __repr__(self):
        return "/"
        
class inst5(object):
    num_childs = 1
    inst_type = "scaler"
    
    def __call__(self, *arg):
        
        global multi_rank

        def multi_rank(rank_df, date):
            # ascending sort
            temp_df = rank_df[rank_df['Date'] == date].sort_values(by=['signals'], ascending=True, inplace=False) 
            temp_df.loc[:,'rank'] = [(i+1)/len(temp_df) for i in range(len(temp_df))]
            return temp_df[['Date', 'Code_', 'rank']]


        def rank(signals, data, num_processes):
            data.loc[:,'signals'] = signals
            rank_df = data[['Date', 'Code_', 'signals']]
            dates = np.unique(data['Date'].values)
            elems = [(rank_df, date) for date in dates]
            pool = multiprocessing.Pool(processes=num_processes)
            rank_df_list = pool.starmap(multi_rank, elems)
            pool.close()
            pool.join()
            
            data.drop(columns=['signals'], inplace=True)  
            temp_df = pd.concat(rank_df_list, ignore_index=True)
            signals = pd.merge(data, temp_df, how='left', on=['Date', 'Code_'])['rank'].to_numpy()

            del [[temp_df, rank_df]]
            gc.collect()

            return signals

        return rank(arg[0], arg[1], arg[2])

    def __repr__(self):
        return "rank"
    
class inst6(object):
    num_childs = 1
    inst_type = "ts_function"
    
    def __call__(self, *arg):
        
        global multi_z_score_r
        
        def multi_z_score_r(z_df, code, window, min_periods=1):
            temp_df = z_df[z_df['Code_'] == code]
            x = temp_df['signals']
            r = x.rolling(window=window, min_periods=min_periods)
            m = r.mean()
            std = r.std(ddof=0)
            temp_df.loc[:,'zscore'] = ((x-m)/std).to_numpy()
            return temp_df[['Date', 'Code_', 'zscore']]

        def ts_zscore(signals, window, data, num_processes):
            data.loc[:, 'signals'] = signals
            temp_df = pd.DataFrame()
            z_df = data[['Date', 'Code_', 'signals']]
            codes = np.unique(data['Code_'].values)
            elems = [(z_df, code, window) for code in codes]
            pool = multiprocessing.Pool(processes=num_processes)
            zscore_df_list = pool.starmap(multi_z_score_r, elems)
            pool.close()
            pool.join()
            
            data.drop(columns=['signals'], inplace=True)

            temp_df = pd.concat(zscore_df_list, ignore_index=True)
            signals = pd.merge(data, temp_df, how='left', on=['Date', 'Code_'])['zscore'].to_numpy()    

            del [[temp_df]]
            gc.collect()

            return signals        

        return ts_zscore(arg[0], arg[1], arg[2], arg[3])
    
    def __repr__(self):
        return "ts_zscore"

class inst7(object):
    num_childs = 1
    inst_type = "ts_function"
    def __call__(self, *arg):
        
        global multi_mean_r

        def multi_mean_r(mean_df, code, window, min_periods=1):
            temp_df = mean_df[mean_df['Code_'] == code]
            x = temp_df['signals']
            r = x.rolling(window=window, min_periods=min_periods)
            m = r.mean()
            temp_df.loc[:,'mean'] = m
            return temp_df[['Date', 'Code_', 'mean']]


        def ts_mean(signals, window, data, num_processes):
            data.loc[:, 'signals'] = signals
            temp_df = pd.DataFrame()
            mean_df = data[['Date', 'Code_', 'signals']]
            codes = np.unique(data['Code_'].values)
            elems = [(mean_df, code, window) for code in codes]
            pool = multiprocessing.Pool(processes=num_processes)
            mean_df_list = pool.starmap(multi_mean_r, elems)
            pool.close()
            pool.join()
            
            data.drop(columns=['signals'], inplace=True)

            temp_df = pd.concat(mean_df_list, ignore_index=True)
            signals = pd.merge(data, temp_df, how='left', on=['Date', 'Code_'])['mean'].to_numpy()

            del [[temp_df]]
            gc.collect()

            return signals
        return ts_mean(arg[0], arg[1], arg[2], arg[3])
    
    def __repr__(self):
        return "ts_mean"



class Algorithm(object):
    
    def __init__(self, init_max_height, init_min_height, init_generate_ratio):
        self.init_max_height = init_max_height
        self.init_min_height = init_min_height
        self.init_generate_ratio = init_generate_ratio
        
        # generate
        self.functions = [inst1, inst2, inst3, inst4, inst1, inst2, inst3, inst4, inst5, inst6, inst7]
        self.windows = [3, 5, 10, 20, 60, 120, 250, 500]
        self.prc_features = ['Open', 'High', 'Close', 'Low', 'MCap'] \
                            + ['pC'+str(i) for i in range(1,21)] + ['pO'+str(i) for i in range(1,21)] \
                            + ['pH'+str(i) for i in range(1,21)] + ['pL'+str(i) for i in range(1,21)] \
                            + ['preiod_L5', 'preiod_L10', 'preiod_L20', 'preiod_H5', 'preiod_H10', 'preiod_H20' ]\
                            + ['Volume']
                            #+ ['ma5', 'ma10', 'ma20', 'ma60', 'maV5', 'maV10', 'maV20', 'maV60']
        
        self.jam_features = ['assets', 'liabilities', 'equity', 'netSales', 'opProfit', 'netProfit', 'costOfSales', \
                             'grossMargin', 'nOpRevenue', 'nOpCost', 'beforeNetProfit', 'stks']

        self.features = self.prc_features + self.jam_features
        
        self.root = None
        
    def __repr__(self):
        def prefix(node):
            ret = ""
            if node != None:
                if node.is_leaf:
                    ret += "{ "+repr(node.value()) # + repr(node.is_leaf)
                    ret += " ( "+str(node.left) + " ) "
                    ret += " ( "+str(node.right)+ " ) "
                    ret += " }"
                else:
                    ret += "{ "+ repr(node.value()) # + repr(node.is_leaf)
                    ret += " ( "+prefix(node.left) + " ) "
                    if node.value.inst_type == 'scaler' or node.value.inst_type == 'ts_function':
                        ret += " ( "+ str(node.right) + " ) "
                    else:
                        ret += " ( "+prefix(node.right)+ " ) "
                    ret += " }"
            else:
                ret += "{ " + "None" + " }"
            
            return ret
        return prefix(self.root)
    
    def create_node(self):
        
        inst = random.choice(self.functions)
        right = None
        if inst.inst_type == 'ts_function':
            right = random.choice(self.windows)

        return node(value = inst, left=None, right=right, is_leaf=False)

    def algorithm_init(self):
        
        def generate(node, height):
            
            inst = node.value
            
            if height >= self.init_max_height:  # stop generate
                node.is_leaf = True
                if inst.inst_type == 'scaler':    
                    node.left = random.choice(self.features)
                    node.right = None
                elif inst.inst_type == 'ts_function':
                    node.left = random.choice(self.features)
                    node.right = random.choice(self.windows)
                elif inst.inst_type == 'function':
                    node.left = random.choice(self.features)
                    node.right = random.choice(self.features)
                else:
                    print('------- type error -------')
                    
            elif height < self.init_min_height:
                if inst.inst_type == 'function': 
                    node.left = self.create_node()
                    node.right = self.create_node()
                    generate(node.left, height+1)
                    generate(node.right, height+1)
                elif inst.inst_type == 'scaler': 
                    node.left = self.create_node()
                    node.right = None
                    generate(node.left, height+1)
                elif inst.inst_type == 'ts_function': 
                    node.left = self.create_node()
                    node.right = random.choice(self.windows)
                    generate(node.left, height+1)
                    

            else:
                # generate with the probability of 50% 
                if random.random() < self.init_generate_ratio:
                    if inst.inst_type == 'function': 
                        node.left = self.create_node()
                        node.right = self.create_node()
                        generate(node.left, height+1)
                        generate(node.right, height+1)
                    elif inst.inst_type == 'scaler': 
                        node.left = self.create_node()
                        node.right = None
                        generate(node.left, height+1)
                    elif inst.inst_type == 'ts_function': 
                        node.left = self.create_node()
                        node.right = random.choice(self.windows)
                        generate(node.left, height+1)
                # stop generate
                else: 
                    node.is_leaf = True
                    if inst.inst_type == 'scaler':    
                        node.left = random.choice(self.features)
                        node.right = None
                    elif inst.inst_type == 'ts_function':
                        node.left = random.choice(self.features)
                        node.right = random.choice(self.windows)
                    elif inst.inst_type == 'function':
                        node.left = random.choice(self.features)
                        node.right = random.choice(self.features)
                    else:
                        print('------- type error -------')
                    
        self.root = self.create_node()
        generate(self.root, 1)
    
    def get_all_nodes(self):
        
        def get_nodes(node):
            lists = [node]

            if node.is_leaf:
                return lists
            else:
                if node.value_type == 'function':
                    lists += get_nodes(node.left)
                    lists += get_nodes(node.right)
                elif node.value_type == 'scaler':
                    lists += get_nodes(node.left)
                elif node.value_type == 'ts_function':    
                    lists += get_nodes(node.left)
                return lists

        return get_nodes(self.root)
    
    def tree_height(self):
        
        def calc_height(node):
            if node.is_leaf:
                return 1
            else:
                left_height = calc_height(node.left)+1
                right_height = 1
                if node.value.inst_type == 'function':
                    right_height = calc_height(node.right)+1
                return max(left_height, right_height)            
            
        return calc_height(self.root)
    
    # crossover
    def xover(self, other):
        
        a1_list = self.get_all_nodes()
        a2_list = other.get_all_nodes()
        
        node1 = random.choice(a1_list)
        node2 = random.choice(a2_list)
        
        try_xover = 0
        while((node1.is_leaf != node2.is_leaf) and try_xover < 5):
            node1 = random.choice(a1_list)
            node2 = random.choice(a2_list)
            try_xover += 1
        
        if try_xover == 5:
            print('-------------- xover failed --------------')
            return 0
        
        if node1.value_type in ['scaler', 'ts_function'] and node2.value_type in ['scaler', 'ts_function']:
            temp = node1.left
            node1.left = node2.left
            node2.left = temp
        
        elif node1.value_type not in ['scaler', 'ts_function'] and node2.value_type in ['scaler', 'ts_function']:
            if random.random() > 0.5:
                temp = node1.left
                node1.left = node2.left
                node2.left = temp
            else:
                temp = node1.right
                node1.right = node2.left
                node2.left = temp
        
        elif node1.value_type in ['scaler', 'ts_function'] and node2.value_type not in ['scaler', 'ts_function']:
            if random.random() > 0.5:
                temp = node1.left
                node1.left = node2.left
                node2.left = temp
            else:
                temp = node1.left
                node1.left = node2.right
                node2.right = temp
        else:
            p = random.random()
            if p < 0.25:
                temp = node1.left
                node1.left = node2.left
                node2.left = temp
            elif p < 0.5:
                temp = node1.left
                node1.left = node2.right
                node2.rightt = temp
            elif p < 0.75:
                temp = node1.right
                node1.right = node2.left
                node2.left = temp
            else:
                temp = node1.right
                node1.right = node2.right
                node2.right = temp
            
        return 0
    
    # mutation
    def mutation(self, mutation_rate):
        
        def mutate(node):
            # mutate leaf node with 1% probability
            mut = 0
            inst = node.value
            if inst.inst_type == 'function': 
                if node.is_leaf:
                    if random.random() < mutation_rate:
                        left = random.choice(self.features)
                        mut += 1
                    if random.random() < mutation_rate:
                        right = random.choice(self.features)
                        mut += 1
                    return mut
                else:
                    return mutate(node.left) + mutate(node.right)
            
            if inst.inst_type == 'scaler':
                if node.is_leaf:
                    if random.random() < mutation_rate:
                        left = random.choice(self.features)
                        mut += 1
                    return mut
                else:
                    return mutate(node.left)
            
            if inst.inst_type == 'ts_function':
                if random.random() < mutation_rate:
                    right = random.choice(self.windows)
                    mut += 1
                
                if node.is_leaf:
                    if random.random() < mutation_rate:
                        left = random.choice(self.features)
                        mut += 1
                    return mut
                else:
                    return mutate(node.left) + mut             
            
        return mutate(self.root)
    
    def fitness(self, data, start_date, calc_num_processes):
        
        def Calc(node):
            inst = node.value()
            if inst.inst_type == 'function':
                if node.is_leaf:
                    return inst(data[node.left].to_numpy(), data[node.right].to_numpy())
                else:
                    left = Calc(node.left)
                    right = Calc(node.right)
                    return inst(left, right)
            
            if inst.inst_type == 'scaler':
                if node.is_leaf:
                    return inst(data[node.left].to_numpy(), data, calc_num_processes)
                else:
                    left = Calc(node.left)
                    return inst(left, data, calc_num_processes)
            
            if inst.inst_type == 'ts_function':
                if node.is_leaf:
                    return inst(data[node.left].to_numpy(), node.right, data, calc_num_processes)
                else:
                    left = Calc(node.left)
                    right = node.right
                    return inst(left, right, data, calc_num_processes)            

        # rebalancing and calculate the profit
        
        signals = Calc(self.root)
        data.loc[:,'signal'] = signals
        
        data_ = data.loc[data['Date'] >= start_date, :]
        day_returns = []
        num_day_signals = []
        for d, group_df in data_.groupby('Date'):
            group_df = group_df.replace([np.inf, -np.inf], np.nan)
            group_df.dropna(subset=['signal'], inplace=True)
            neut = group_df['signal'] - group_df['signal'].mean()
            weight = (neut/neut.abs().sum()).to_numpy()
            day_return = np.sum(weight * (group_df['nextC1/C'].to_numpy()-1))
            day_returns.append(day_return)
            num_day_signals.append(len(group_df))
        data.drop(columns=['signal'], inplace=True)        
            
        return day_returns, num_day_signals
    
    
        
   

