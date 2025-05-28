# PACKAGES
import numpy as np
import matplotlib.pyplot as plt
import itertools
from datetime import datetime
from glob import glob
from IPython.display import clear_output

# CONSTATNST
FS=12 # fontsize for plots
READ_LENGHT=252 
PROMOTER="TCTTGTGGAAAGGACGAAACACCG"
TERMINATOR="GTTTAAGAGCTATGCTGGAAACAG"
MAX_READ_LENGHT=READ_LENGHT+len(PROMOTER)+len(TERMINATOR)


OV354_BPK_CLONE_F1="CCTGACGTCGCTAGCTGTAC"
OV354_BPK_CLONE_R1="TTTGCTGGCCTTTTGCTCAC"
LONG_AMPLICON_SEQ_LEFT="CCTGACGTCGCTAGCTGTACAAAAAAGCAGGCTTTAAAGGAACCAATTCAGTCGACTGGATCCGGTACCAAGGTCGGGCAGGAAGAGGGCCTATTTCCCATGATTCCTTCATATTTGCATATACGATACAAGGCTGTTAGAGAGATAATTAGAATTAATTTGACTGTAAACACAAAGATATTAGTACAAAATACGTGACGTAGAAAGTAATAATTTCTTGGGTAGTTTGCAGTTTTAAAATTATGTTTTAAAATGGACTATCATATGCTTACCGTAACTTGAAAGTATTTCGATTTCTTGGCTTTATATA"
LONG_AMPLICON_SEQ_RIGHT="TAAAATAAGGCTAGTCCGTTATCAACTTGAAAAAGTGGCACCGAGTCGGTGCTTTTTTTAAGCTTGGGCCGCTCGAGGTACCTCTCTACATATGACATGTGAGCAAAAGGCCAGCAAA"

# FUNCTIONS
def create_fasta_from_reference(input_file, output_file, add_before='', add_after=''):
    with open(input_file, 'r') as f_in, open(output_file, 'w') as f_out:
        seq_num = 1
        for line in f_in:
            line = line.strip()
            if line:  # skip empty lines
                f_out.write(f">sequence{seq_num}\n")
                f_out.write(f"{add_before + line + add_after}\n")
                seq_num += 1

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
    ''' The frequency of possible k-mers in a given sequence in descending order. '''
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
            if count[i][1] !=0: print(f'  {count[i][0]} - {count[i][1]}')
        print('\\end{run_chech()}')
    else:
        return count  
    return None

# -----------------------------------------------------------------------------------------------------------------------------------------
### How to generate the reads?
# Create a slots (`np.array`) for reads.
# If the given hexamer does not exists in any reads, then append it somewhere with the following rules:
# - Append the hexamer to the read where the overlap is the biggest. 
#     - If more than one exists append the shortest read.
# - If there is no overlap anywhere, then 
#     - append the hexamert to a new read (if it is possible) OR
#     - append to the shorthest existing read

def fill_reads_with_k_mers(reads, k_mers, min_overlap, num_of_while_loops, print_infos):
    ''' Help function for generate_reads_v3()'''
    # an empty list for the skipped k-mers
    skipped_k_mers = []
    
    for i, k_mer in enumerate(k_mers):
            
        # if the k-mer does not exists in any read, then put append it using the rules.
        if not any(k_mer in read for read in reads):
                        
            overlaps_in_reads = []
            # Iterate through all reads
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
                too_long_reads = np.where((np.vectorize(len)(reads) + (len(k_mer)-np.max(overlaps_in_reads))) >= READ_LENGHT)[0]    
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
        if print_infos:
            #clear_output(wait=True)
            print(f'{((i+1)/len(k_mers)*100):.0f}% finished, {num_of_while_loops} while loop(s) remaining.', end='\r')
              
    return reads, skipped_k_mers


def generate_reads(k_mers, num_of_reads, min_overlap=None, max_while_counter=20, print_infos=True):
    ''' Generate reads '''
    # the result will be stored in 'reads' var. as a numpy array
    reads = np.empty(num_of_reads, dtype='U300')
    # fill all reads with an initial k-mer
    reads[:num_of_reads] = k_mers[:num_of_reads]
    skipped_k_mers = k_mers[num_of_reads:].copy()

    if min_overlap == None:
        min_overlap = len(k_mers[0]) // 2

    while_counter = 0
    while max_while_counter >= while_counter:
        k_mers_len_before = len(skipped_k_mers)
        reads, skipped_k_mers = fill_reads_with_k_mers(reads, skipped_k_mers, min_overlap, max_while_counter-while_counter, print_infos)
        k_mers_len_after = len(skipped_k_mers)
        if len(skipped_k_mers) == 0:
            if print_infos:
                print(f'No more skipped k-mer after {while_counter} loop. {20*" "}')
            break
        if k_mers_len_before == k_mers_len_after:
            if print_infos:
                print(f'Cannot append more k-mers after {while_counter} loop. {20*" "}')
            break 
        while_counter+=1        
        
    if any(np.vectorize(len)(reads) > READ_LENGHT):
        print('Something went wrong: there is a too long read.')
        return None
    if print_infos:
        print(f'Number of skipped k-mers: {len(skipped_k_mers)}  {20*" "}')
    if sum((READ_LENGHT - np.vectorize(len)(reads)) // len(k_mers[0])) < len(skipped_k_mers):
        print("Error can be occured: skipped k-mers cannot be appended to reads --> increase the 'num_of_read' ")

    return list(reads), np.array(skipped_k_mers)    


def append_skipped_k_mers(reads, skipped_k_mers):
    ''' Append skipped k-mers to reads '''
    k_mer_idx = 0
    reads_with_skipped_k_mers = []
    for i, read in enumerate(reads):
        reads_with_skipped_k_mers.append(read)
        while k_mer_idx < len(skipped_k_mers):
            if len(reads_with_skipped_k_mers[i]) + len(skipped_k_mers[k_mer_idx]) <= READ_LENGHT:
                reads_with_skipped_k_mers[i] += skipped_k_mers[k_mer_idx]
                k_mer_idx +=1
            else:
                break
    return reads_with_skipped_k_mers


def fill_reads_to_READ_LENGHT(reads, kmers):
    ''' Extend reads to READ_LENGHT bp using rare k-mers, then fill with random bases '''
    result = []
    kmers_to_append = kmers_sorted_by_frequency(reads, kmers)[:, 0]  # Get sorted k-mers by frequency
    kmer_counter = 0
    four_bases = ['A', 'C', 'T', 'G']
    for read in reads:
        # phase I. Extend with rare k-mers
        while len(read) + len(kmers_to_append[kmer_counter]) <= READ_LENGHT:
            read += kmers_to_append[kmer_counter]
            kmer_counter += 1
        # phase II. Fill remaining length with random bases
        while len(read) < READ_LENGHT:
            read += np.random.choice(four_bases)
        result.append(read)
    return result

def add_limit_sequences(reads):
    result=[]
    for seq in reads:
        result.append(PROMOTER + seq + TERMINATOR)
    return result

# -----------------------------------------------------------------------------------------------------------------------------------------

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
    plt.xlim(0, np.round(MAX_READ_LENGHT, -1))
    plt.ylim(0,1)
    plt.hlines(0.6, 0, np.round(MAX_READ_LENGHT, -1), color='r')
    plt.hlines(0.4, 0, np.round(MAX_READ_LENGHT, -1), color='r')
    plt.title(f'Cumulative G-C ratio\n{pct*100:.2f}% of bases in the tolerance interval', fontsize=FS+3)
    plt.ylabel('G-C content', fontsize=FS)
    plt.xlabel('read index', fontsize=FS)
    plt.grid()
    return None


def gc_cont_like_gpt_code(sequence):
    gc_content = []
    for seq in sequence:
        gc_content.append( (seq.count('G') + seq.count('C'))/len(seq) )
    return np.round(np.array(gc_content)*100, 1)


def plot_GC_content(gc_con, title=''):
    plt.figure(figsize=(8,6))
    plt.plot(gc_con, 'o', c='b')
    plt.title(title, fontsize=FS+3)
    plt.xlabel('read id', fontsize=FS)
    plt.ylabel('GC content [%]', fontsize=FS)
    plt.grid()
    plt.show()
    return None


def found_in_data(generated_params, save_or_load=''):
    files = glob('data/*')
    existing_parameters=[]
    for file in files:
        existing_parameters.append([int(part) for el in file.split('_')[1:] for part in el.split('-') if part.isdigit()])
    if save_or_load=='save':
        return generated_params in existing_parameters
    elif save_or_load=='load':
        return files[existing_parameters.index(generated_params)]
    else:
        assert(f"save_or_load must be 'save' or 'load' not '{save_or_load}'.")

        
def save_to_txt(sequences, path='', default_params=[]):
    if not found_in_data(default_params, 'save'):
        if not path:
            path = f"data/{datetime.now().strftime('%m%d-%H%M')}_"+\
                   f"{default_params[0]}-mer_"+\
                   f"{default_params[1]}-reads_"+\
                   f"{default_params[2]}-min-overlap.txt"
        with open(path, 'w') as file:
            for seq in sequences:
                file.write(f"{seq}\n")
        print(f"Data saved to '{path}'")
    else:
        print("A data file exists with these parameters:")
        print(f"k={default_params[0]}, num_of_reads={default_params[1]}, min_overlap={default_params[2]}")
    return None

def save_to_excel(sequences, file_name):
    import pandas as pd
    seq_ids = [f"sequence_{i + 1}" for i in range(len(sequences))]
    df = pd.DataFrame({'Sequence ID': seq_ids, 'Sequence': sequences})
    df.to_excel(file_name, index=False)
    return None


def load_from_txt(params):
    try: 
        with open(found_in_data(params, 'load'), 'r') as file:
            return [line.strip() for line in file]
    except:
        raise ValueError("The data file does not exist with these parameters: "+\
                        f"k={params[0]}, num_of_reads={params[1]}, min_overlap={params[2]}")




