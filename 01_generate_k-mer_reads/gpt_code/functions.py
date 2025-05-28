# packages
import numpy as np
import matplotlib.pyplot as plt
import itertools

# fontsize for plots
FS=12

# functions
def combine_strings(string):
    ''' Add strings to each other. '''
    return ''.join(string)


def generate_k_mer_list(possible_bases, k_mer):
    ''' All possible repeated variations of k-mers. '''
    itr_prod = itertools.product(possible_bases, repeat=k_mer)
    k_mer_list = []
    for i in range(len(possible_bases)**k_mer):
        k_mer_list.append(combine_strings(next(itr_prod)))
    return np.array(k_mer_list)


def kmers_sorted_by_frequency(chains, k_mers, order_desc=False):
    ''' The frequency of possible k-mers in a given sequence in a given order. '''
    numbers = {}
    for element in k_mers:
        n=0
        for chain in chains:
            n += chain.count(element)
            numbers[element] = n
    sort_numbers = np.array(sorted(numbers.items(), key=lambda x:x[1], reverse=order_desc))
    return sort_numbers


def check_frequency(chains, k_mers, print_count=True):
    ''' Counting identical frequencies --> how many k-mers appear n times in a sequence. '''
    sort_numbers = kmers_sorted_by_frequency(chains, k_mers, order_desc=True)
    count = []
    for i in range(1, int(sort_numbers[0,1])+1):
        count.append([i, sum(np.array(sort_numbers[:,1], dtype=int) == i)])
    # missing elemets 
    count.append([0, sum(np.array(sort_numbers[:,1], dtype=int) == 0)])
    if print_count:
        print('\\begin{run_chech()}')
        print('  frequency - the number of k-mers:')
        for i in range(len(count)):
            # print if element type is not missing
            if count[i][1] !=0:
                print(f'  {count[i][0]} - {count[i][1]}')
        print('\end{run_chech()}')
    else:
        return np.array(count)  
    return None

