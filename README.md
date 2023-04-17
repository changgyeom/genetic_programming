# Portfolio Optimization Using Genetic Programming


## genetic programming framework

1. init tree algorithms
2. evaluate tree algorithms
3. make new generation
  - selection
  - subtree crossover
  - mutation
  - replace
4. repeat 2, 3 until satisfying terminal condition
<p align="center">
  <img src="https://user-images.githubusercontent.com/43362326/230775295-f8203acb-9dea-42a0-ac38-6327290c0ddf.png" width="70%" height="70%">
</p>


## tree_algorithm.py
- define tree algorithm
- tree algorithm is composed of function sets and terminal sets
- the example of tree algorithm

<p align="center">
  <img src="https://user-images.githubusercontent.com/43362326/230756753-970637e8-21ab-43a1-8871-ff1a541b7efc.png" width="50%" height="50%">
</p>


## train.py
- make tree algorithms 
- train tree algorithms using genetic programming
- argument information 
  - init : make new initial chromosomes
  - start_date, end_date : period of training data
  - N : the number of chromosomes
  - k : the number of replaced chromosomes in one cycle
  - init_max_h : the max height of initial chromosomes
  - train_max_h : the max height of chromosomes in training cycle
  - num_processes : the number of using cpu
  - sub_num_processes : the number of using cpu when calculating the values of initial chromosomes
  - calc_num_processes : the number of using cpu when calculating the values of chromosomes in training cycle
  - exp_num : the prefix of saved file name
- commend example
```

  python3 train.py --init=True --start_date=20100000 --end_date=20130000 --N=5000 --k=200 --init_max_h=4 --train_max_h=8 --num_processes=10 --sub_num_processes=4 --calc_num_processes=10 --exp_num=1_1_
  
```
  
  
  


