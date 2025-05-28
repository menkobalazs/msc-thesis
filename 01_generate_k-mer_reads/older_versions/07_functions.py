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


def shortest_chain(k_mers): ## --> ChatGPT code, promt: given the following array; create the shortest string which contains all element of the given array
    '''
    With the help of ChatGPT, the shortest string that contains all possible hexamers,
    extending it only as much as necessary so that if a given k-mer already exists in the chain, it is not added again.
    '''
    result = k_mers[0]
    for i in range(1, len(k_mers)):
        overlap = len(k_mers[i]) - 1
        # If the k-mer already exists in the word chain, do not append it to the result.
        if result.count(k_mers[i]) == 0: # added to chatGPT code
            while overlap >= 0: # Backtracking until the k-mer matches the end of the chain.
                if result.endswith(k_mers[i][:overlap]):
                    break
                overlap -= 1
            result += k_mers[i][overlap:]
    return result


def kmers_sorted_by_frequency(chains, k_mers):
    ''' The frequency of possible k-mers in a given sequence in descending order. '''
    numbers = {}
    for element in k_mers:
        n=0
        for chain in chains:
            n += chain.count(element)
            numbers[element] = n
    sort_numbers = np.array(sorted(numbers.items(), key=lambda x:x[1], reverse=True))
    return sort_numbers


def check_frequency(chains, k_mers, print_count=True):
    ''' Counting identical frequencies --> how many k-mers appear n times in a sequence. '''
    sort_numbers = kmers_sorted_by_frequency(chains, k_mers)
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
        return count  
    return None


def calculate_GC_content(reads, tol=0.1):
    ''' Calculate the GC content trought reads '''
    gc_content_per_index_for_all_reads = []
    pct_in_tol_intervall = []
    for read in reads:
        gc_content_per_index = []
        for i in range(1, len(read) + 1):
            gc_count = read[:i].count('G') + read[:i].count('C')  # Count G and C
            gc_content_per_index.append(gc_count / i )
        gc_content_per_index_for_all_reads.append(gc_content_per_index)
        in_tolerance_interval = (np.array(gc_content_per_index) >= (0.5-tol)) & (np.array(gc_content_per_index) <= (0.5+tol))
        pct_in_tol_intervall.append( sum(in_tolerance_interval) / len(gc_content_per_index) )
    return np.array(gc_content_per_index_for_all_reads, dtype='object'), np.array(pct_in_tol_intervall)

    
def plot_gc_ratio_for_a_read(read_GC_avg, pct):
    ''' Plot the G-C ratio trought a read '''
    plt.figure(figsize=(8,6))
    plt.plot(read_GC_avg, 'o--', c='blue', ms=10)
    plt.xlim(0,240)
    plt.ylim(0,1)
    plt.hlines(0.6, 0, 240, color='r')
    plt.hlines(0.4, 0, 240, color='r')
    plt.title(f'G-C ratio in a given read\n{pct*100:.2f}% of bases in the interval', fontsize=FS+3)
    plt.ylabel('G-C ratio', fontsize=FS)
    plt.xlabel('read index', fontsize=FS)
    plt.grid()
    return None

# --------------------------------------------------------------------------------------------------------------------------------
### How to generate the reads?
# Create a slots (`np.array`) for reads.
# If the given hexamer does not exists in any reads, then append it somewhere with the following rules:
# - Append the hexamer to the read where the overlap is the biggest. 
#     - If more than one exists append the shortest read.
# - If there is no overlap anywhere, then 
#     - append the hexamert to a new read (if it is possible) OR
#     - append to the shorthest existing read

def fill_reads_with_k_mers(reads, k_mers, min_overlap, num_of_while_loops):
    ''' Help function for generate_reads_v3()'''
    # an empty list for the skipped k-mers
    skipped_k_mers = []
    
    for i, k_mer in enumerate(k_mers):
            
        # if the k-mer does not exists in any read, then put append it using the rules.
        if not any(k_mer in read for read in reads):
                        
            overlaps_in_reads = []
            # Iterate trough all reads
            for j in range(len(reads)):   
                # the number of possible overlapping bases
                overlap = len(k_mer) - 1
                # backtracking until the k-mer matches to the end of read j REWRITE
                while overlap >= 0: 
                        # if the read end with the same character(s) as the hexamer begin.
                        if reads[j].endswith(k_mer[:overlap]):
                            # save the number of overlapping characters
                            overlaps_in_reads.append(overlap)
                            # break the while loop
                            break
                        # there is no overlap with n character --> try with n-1
                        overlap -= 1
        
            # if all overlaps are less then or equal to the minimal overlap 
            if np.all(np.array(overlaps_in_reads) <= min_overlap):
                # skip the k-mer
                skipped_k_mers.append(k_mer)                                  
            # else there is/are overlap(s) somewhere 
            else:          
                # find the indices of maximal overlaps
                max_overlaps_indices = np.where(overlaps_in_reads == np.max(overlaps_in_reads))[0]
                # find too long reads 
                too_long_reads = np.where((np.vectorize(len)(reads) + max(overlaps_in_reads)) >= 239)[0]    
                # use only those indices what are not too long
                selected_reads = np.setdiff1d(max_overlaps_indices, too_long_reads)
                
                # if more than one minimal overlap exists
                if len(selected_reads) > 1:                   
                    # get the length of selected reads
                    length_of_selected_reads = np.vectorize(len)(reads[selected_reads])
                    # find the indices of the minimal read lenght
                    min_read_len_idx = np.where(length_of_selected_reads == np.min(length_of_selected_reads))[0]
                    # random seed for reproduction
                    np.random.seed(137)
                    # append the hexamer (w/o overlap) to one of the shortest read                  
                    reads[selected_reads[np.random.choice(min_read_len_idx)]] += k_mer[max(overlaps_in_reads):]
                    
                # else one minimal overlap exists
                elif len(selected_reads) == 1:
                    #append the hexamer (w/o overlap)
                    reads[selected_reads[0]] += k_mer[max(overlaps_in_reads):]
                else:
                    # skip the k-mer
                    skipped_k_mers.append(k_mer)
     
        # print 
        print(f'{((i+1)/len(k_mers)*100):.0f}% finished, {num_of_while_loops} while loop(s) remaining. {10*" "}', end='\r')
              
    return reads, skipped_k_mers


def generate_reads_v3(k_mers, num_of_reads, min_overlap=None, max_while_counter=5, print_anything=True):
    ''' Generate reads '''
    # the result will be stored in 'reads' var. as a numpy array
    reads = np.empty(num_of_reads, dtype='U300')
    
    # fill all reads with an initial k-mer
    reads[:num_of_reads] = k_mers[:num_of_reads]
    
    skipped_k_mers = k_mers[num_of_reads:].copy()

    if min_overlap == None:
        min_overlap = len(k_mers[0]) // 2
        print(f'minimum overlap set to {min_overlap}')

    while_counter = 0
    
    #k_mers_copy = k_mers.copy()
    while max_while_counter >= while_counter:
        k_mers_len_before = len(skipped_k_mers)
        reads, skipped_k_mers = fill_reads_with_k_mers(reads, skipped_k_mers, min_overlap, max_while_counter-while_counter)
        k_mers_len_after = len(skipped_k_mers)
        
        if len(skipped_k_mers) == 0:
            if print_anything:
                print(f'No more skipped k-mer after {while_counter} loop. {20*" "}')
            break
        if k_mers_len_before == k_mers_len_after:
            if print_anything:
                print(f'Cannot append more k-mers after {while_counter} loop. {20*" "}')
            break 
        
        while_counter+=1        
        
    if print_anything:
        print(f'Number of skipped k-mers: {len(skipped_k_mers)}  {20*" "}')
    if any(np.vectorize(len)(reads) > 240) & print_anything:
        print('Something went wrong: there is a too long read.')

    return list(reads), np.array(skipped_k_mers)    



def append_skipped_k_mers(reads, skipped_k_mers):
    ''' Append skipped k-mers to the stast of the worst scored reads '''
    # needs to write
    return reads




# -------------------------------------------------------------------------------------------------------------------------------------------------

def run_seed_simulation(start, end):
    means, stdevs = [], []
    for i in range(start, end):

        hexamers = generate_k_mer_list('ACTG', 6)
        np.random.seed(i)
        np.random.shuffle(hexamers)
        num_of_reads, min_overlap = 27, 2
        reads, _ = generate_reads_v3(hexamers, num_of_reads=num_of_reads, min_overlap=min_overlap, max_while_counter=20, print_anything=False)
        _, pct_in_tol_intervall = calculate_GC_content(reads)
        means.append(np.mean(pct_in_tol_intervall))
        stdevs.append(np.std(pct_in_tol_intervall))
        
    print(f'The score of simulation #{start+np.argmax(means)} = ({max(means)*100:.5f} +/- {stdevs[np.argmax(means)]*100:.5f}) %')
    return max(means)
