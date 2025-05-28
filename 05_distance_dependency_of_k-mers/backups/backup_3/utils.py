import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm

FS = 12 # fontsize

# ======================================================================================================
# Handle .fast5 files

def read_fast5(fname, single_or_multi_read, get_read_ids=False):
    'read only one fast5 file'
    from ont_fast5_api.fast5_interface import get_fast5_file
    def raw_to_current(rawdata, offset, range_, digitisation):
        return list((np.array(rawdata) + offset) * range_ / digitisation)
    
    if single_or_multi_read not in ['s', 'm']:
        raise ValueError('single_or_multi variable must be "s" or "m".')
    data = []
    read_ids = []
    with get_fast5_file(fname, mode="r") as f:
        for read in f.get_reads():
            ch=read.get_channel_info()
            data.append(raw_to_current(read.get_raw_data(), ch['offset'], ch['range'], ch['digitisation']))
            read_ids.append(read.read_id)
    if single_or_multi_read == 's':
        if get_read_ids:
            return np.array(data[0]), read_ids
        return np.array(data[0]) # single read --> dim=1
    elif single_or_multi_read == 'm':
        if get_read_ids:
            return np.array(data, dtype='object'), read_ids
        return np.array(data, dtype='object')

def find_read_id_index(read_id_list, searched_read_id):
    read_id_list = np.array(read_id_list)
    index = np.where(np.char.find(read_id_list, searched_read_id) != -1)[0]
    if len(index) == 1:
        return index[0]
    print(f'Error: no or more indices found; len(index)={len(index)}')
    return None

def split_raw_signal(raw_signal, move_table, stride):
    start_of_bases = (np.where(move_table == 1)[0]) * stride
    signals = [raw_signal[i:j] for i, j in zip(start_of_bases, start_of_bases[1:])]
    mean_signals = [np.mean(raw_signal[i:j]) for i, j in zip(start_of_bases, start_of_bases[1:])]
    return start_of_bases, signals, mean_signals

def replace_T_to_U(data):
    if  type(data) not in [list, np.ndarray]:
        raise TypeError(f'sam_file_names must be a list')
    for record in tqdm(data, total=len(data), desc='Replacing'):
        record['sequence'] = np.array(['U' if base == 'T' else base for base in record['sequence']])
    return np.array(data)

# ======================================================================================================
# Handle .sam files

def get_data(sam):
    pattern = re.compile(r'(\d+)([MIDNSHP=X])')
    cigar_str = np.array(pattern.findall(sam[5])).T
    mv_table = np.fromstring(sam[21].split("[")[1].split("]")[0], sep=',', dtype=int)
    return  {
        'read_id':sam[0], 
        'sam_flag':int(sam[1]),
        'mapped_position_index':int(sam[3])-1,
        'map_quality':int(sam[4]),
        'cigar_str':(cigar_str[0].astype(int), cigar_str[1]),
        'sequence':np.array(list(sam[9])),
        'move_table':mv_table[1:],
        'stride':mv_table[0],
        'trim_offset':int(sam[29][3:]),
        'fast5_file_name':sam[27][3:]
       } 

def check_indel_frequency(cigar_data, max_insertion, max_deletion, seq_length, max_indel_frequency):
    cigar_data = np.column_stack(cigar_data)
    number_of_I = cigar_data[cigar_data[:,1] == 'I'][:,0].astype(int) if 'I' in cigar_data[:, 1] else [0]
    number_of_D = cigar_data[cigar_data[:,1] == 'D'][:,0].astype(int) if 'D' in cigar_data[:, 1] else [0]
    # max indel value reached:
    if np.max(number_of_I)>=max_insertion:
        return False
    if np.max(number_of_D)>=max_deletion:
        return False
    # too high indel frequency
    total_M = np.sum(cigar_data[cigar_data[:,1] == 'M'][:,0].astype(int))
    if (np.sum(number_of_D)+np.sum(number_of_I))/seq_length >= max_indel_frequency:
        return False
    else:
        return True


def read_sam(sam_file_names, min_length=0, max_length=1e5, min_MAPQ=0, wrong_flags={4, 256, 272, 2048, 2064}, 
             max_insertion=10, max_deletion=10, max_indel_frequency=0.4, verbose=False):
    ''' read sam file '''
    ### For more files
    if  type(sam_file_names) == list:
        all_sam_data = []
        for sam_file_name in tqdm(sam_file_names, total=len(sam_file_names), desc='Loading data'):
            with open(sam_file_name, 'r') as file:
                sam_file = file.read().split('\n')[:-1]
                sam_file = [line for line in sam_file if not line.startswith('@')]
            one_sam_data = []
            for line in sam_file:
                sam = line.split('\t')
                # There are cases, when tags (or the whole sequence) are missing (or in a few cases one line added.)
                if len(sam)==31:
                    data = get_data(sam)
                    if (len(data['sequence']) <= max_length and
                        len(data['sequence']) >= min_length and
                        data['map_quality'] >= min_MAPQ and
                        data['sam_flag'] not in wrong_flags and
                        check_indel_frequency(data['cigar_str'], max_insertion, max_deletion, len(data['sequence']), max_indel_frequency)
                       ): one_sam_data.append(data)
            all_sam_data.append(np.array(one_sam_data, dtype='object'))
        result = np.hstack(all_sam_data, dtype='object')
        if verbose: print(f"Number of reads stored: {len(result)}")
        return result
        
    ### For only one file
    elif  type(sam_file_names) == str:
        with open(sam_file_names, 'r') as file:
            sam_file = file.read().split('\n')[:-1]
            sam_file = [line for line in sam_file if not line.startswith('@')]
        sam_data = []
        for line in tqdm(sam_file,  total=len(sam_file), desc="Loading file"):
            sam = line.split('\t')
            if len(sam)==31:
                data = get_data(sam)
                if (len(data['sequence']) <= max_length and
                    len(data['sequence']) >= min_length and
                    data['map_quality'] >= min_MAPQ and
                    data['sam_flag'] not in wrong_flags                     
                   ): sam_data.append(data)
        result = np.array(sam_data, dtype='object')
        if verbose: print(f"Number of reads stored: {len(result)}")
        return result
        
    else:
        raise TypeError(f"'sam_file_names' must be a string or list of strings, not {type(sam_file_names)}")

def get_feature_from_sam_data(data, feature):
    return np.array([entry[feature] for entry in data ], dtype='object')

def search_in_data(data, key, element, only_first_match=True, use_tqdm=False):
    if not use_tqdm:
        def tqdm(iterable, total=None, desc=None): 
            return iterable  
    if only_first_match:
        for entry in data:
            if entry[key] == element:
                return entry
        print(f"There is no matching element with '{element}' in '{data}'/{key}.")
        return None
    else:
        result = []
        for entry in tqdm(data, total=len(data), desc="Searching in the data"):
            if entry[key] == element:
                result.append(entry)
        return result
# ======================================================================================================
# Othor functions
def get_raw_signal_with_bases(sam_data, fast5_path, reference_sequence, verbose=False):
    import warnings
    result = []

    for entry in tqdm(sam_data, total=len(sam_data), desc='Processing data'):
        fast5_signals, read_ids = read_fast5(fast5_path + entry['fast5_file_name'], 'm', get_read_ids=True)

        # Get split raw signal
        _, signals, _ = split_raw_signal(fast5_signals[find_read_id_index(read_ids, entry['read_id'])][entry['trim_offset']:], 
                                         entry['move_table'], entry['stride'])

        counter = 0
        ref_counter = entry['mapped_position_index']

        sequence = []
        ref_seq_with_deletions_insertions = []

        signal = []

        for i, (op, length) in enumerate(zip(entry['cigar_str'][1], entry['cigar_str'][0])): 
            if op == "S":  
                counter += length  

            elif op == "M":  
                sequence.extend(entry['sequence'][counter:counter + length])  
                signal.append(signals[counter])
                counter += length  
                
                ref_seq_with_deletions_insertions.extend(reference_sequence[ref_counter:ref_counter + length]) 
                ref_counter += length  

            elif op == "I":  
                sequence.extend(entry['sequence'][counter:counter + length])  
                signal.append(signals[counter])
                counter += length  
                
                ref_seq_with_deletions_insertions.extend([''] * length)  

            elif op == "D":  
                sequence.extend([''] * length)  
                signal.append([None] * length * entry['stride'])  # it means there is no signal due to deletion
                
                ref_seq_with_deletions_insertions.extend(reference_sequence[ref_counter:ref_counter + length])  
                ref_counter += length  
                            
            else:  
                warnings.warn(f"Invalid CIGAR operator: '{op}'", UserWarning)  
                break  

        result.append({
            'sequence': np.array(sequence),
            'ref_seq': np.array(ref_seq_with_deletions_insertions),
            'signal': np.array(signal, dtype='object')
        })

    return result

def test_basecall_accuracy(result_dict):
    pct=[]
    for i in range(len(result_dict)):
        matching, not_matching = 0,0
        for s, r in zip(result_dict[i]['sequence'], result_dict[i]['ref_seq']):
            if not s=='' and not r=='' :
                if s==r:
                    matching +=1 
                else: 
                    not_matching+=1
        
        pct.append(matching/ (matching+not_matching))
    print(f'Min accuracy:  {100*min(pct):.3f}%')
    print(f'Max accuracy:  {100*max(pct):.3f}%')
    print(f'Mean accuracy: {100*np.mean(pct):.3f}%')
    print(f'Std accuracy:  {100*np.std(pct):.3f}%')
    return None


def find_the_loneliest_base_in_seq(seq, base='T', verbose=0):
    def find_the_farthest_neighbors(dst):
        'Squared geometric mean'
        squared_geom_mean = []
        for i in range(len(dst)-1):
            squared_geom_mean.append(dst[i]*dst[i+1] )
        return np.argmax(squared_geom_mean)+1
    
    positions = np.where(seq == base)[0]
    distances = np.diff(positions)
    idx_of_loneliest_base = find_the_farthest_neighbors(distances)
    pos_of_loneliest_base = positions[idx_of_loneliest_base]
    if verbose:
        print(f"The position of the loneliest base '{base}': {idx_of_loneliest_base}")
        print(f"Distances between the neighborhood: {thymine_distances[idx_of_loneliest_thymine-2:idx_of_loneliest_thymine+2]}")
        left_bound  = pos_of_loneliest_base-distances[idx_of_loneliest_base-1]
        right_bound = pos_of_loneliest_base+distances[idx_of_loneliest_base]+1
        print("The subsequence:\n",seq[left_bound:right_bound])
        return None
    else:
        return pos_of_loneliest_base