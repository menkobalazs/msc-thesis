# optimized by ChatGPT o3mini:
# https://chatgpt.com/share/67ed2467-682c-8013-96db-5a9d97032eb4


import numpy as np
from tqdm import tqdm

def read_fast5__optimized(fname, single_or_multi_read, get_read_ids=False):
    """
    Read one fast5 file and return the raw current signal(s) and, optionally, read ids.
    """
    from ont_fast5_api.fast5_interface import get_fast5_file

    def raw_to_current(rawdata, offset, range_, digitisation):
        # Use vectorized conversion for speed
        return (np.asarray(rawdata, dtype=np.float32) + offset) * range_ / digitisation

    if single_or_multi_read not in ['s', 'm']:
        raise ValueError('single_or_multi variable must be "s" or "m".')
    
    data, read_ids = [], []
    with get_fast5_file(fname, mode="r") as f:
        for read in f.get_reads():
            ch = read.get_channel_info()
            data.append(raw_to_current(read.get_raw_data(), ch['offset'], ch['range'], ch['digitisation']))
            read_ids.append(read.read_id)
            
    # Return the first read for 's' or all reads for 'm'
    if single_or_multi_read == 's':
        if get_read_ids:
            return np.array(data[0]), read_ids
        return np.array(data[0])
    else:  # single_or_multi_read == 'm'
        if get_read_ids:
            return np.array(data, dtype='object'), read_ids
        return np.array(data, dtype='object')


def find_read_id_index__optimized(read_id_list, searched_read_id):
    """
    Find the index of the read whose id contains the searched_read_id as a substring.
    Returns the index if exactly one match is found, otherwise prints an error.
    """
    indices = [i for i, rid in enumerate(read_id_list) if searched_read_id in rid]
    if len(indices) == 1:
        return indices[0]
    print(f'Error: no or more indices found; len(indices)={len(indices)}')
    return None


def split_raw_signal__optimized(raw_signal, move_table, stride):
    """
    Split the raw signal into segments defined by the positions where move_table == 1.
    Returns:
      - start_of_bases: starting indices of bases in the raw signal
      - signals: list of signal segments
      - mean_signals: mean of each signal segment
    """
    # Use np.flatnonzero to get indices more directly
    start_of_bases = np.flatnonzero(move_table == 1) * stride
    # Append end index to cover last segment
    end_index = len(raw_signal)
    # Compute segments using zip over start indices and the next start (or end_index)
    signals = [raw_signal[i:j] for i, j in zip(start_of_bases, np.append(start_of_bases[1:], end_index))]
    mean_signals = [np.mean(segment) for segment in signals]
    return start_of_bases, signals, mean_signals


def get_raw_signal_with_bases(sam_data, fast5_path, reference_sequence):
    """
    For each SAM entry, extract the corresponding raw signal segments and
    map them to bases and the reference sequence based on the CIGAR string.
    
    Optimizations include caching of fast5 file reads (by file name) and local variable lookups.
    """
    result = []
    # Cache to avoid re-reading the same fast5 file for multiple entries
    fast5_cache = {}

    for entry in tqdm(sam_data, total=len(sam_data), desc='Processing data'):
        # Construct full path for the fast5 file
        fname = fast5_path + entry['fast5_file_name']
        if fname not in fast5_cache:
            fast5_signals, read_ids = read_fast5__optimized(fname, 'm', get_read_ids=True)
            fast5_cache[fname] = (fast5_signals, read_ids)
        else:
            fast5_signals, read_ids = fast5_cache[fname]

        idx = find_read_id_index__optimized(read_ids, entry['read_id'])
        if idx is None:
            continue  # Skip this entry if read id not found properly

        # Slice raw signal from trim_offset onward and split using move_table
        raw_signal = fast5_signals[idx][entry['trim_offset']:]
        _, signals, mean_signals = split_raw_signal__optimized(raw_signal, entry['move_table'], entry['stride'])


        # Initialize counters and lists
        counter = 0
        ref_counter = entry['mapped_position_index']
        sequence = []
        ref_seq_with_deletions_insertions = []
        signal_segments = []  # This will store the signal corresponding to each operation
        mean_signal_segments = []

        # Cache the CIGAR string parts locally
        cigar_ops = entry['cigar_str'][1]
        cigar_lengths = entry['cigar_str'][0]
        seq = entry['sequence']

            
        for op, length in zip(cigar_ops, cigar_lengths):
            if op == "S":
                counter += length
            elif op == "M":
                sequence.extend(seq[counter:counter+length])
                signal_segments.extend(signals[counter:counter+length])
                mean_signal_segments.extend(mean_signals[counter:counter+length])
                counter += length
                ref_seq_with_deletions_insertions.extend(reference_sequence[ref_counter:ref_counter+length])
                ref_counter += length
            elif op == "I":
                sequence.extend(seq[counter:counter+length])
                # Use extend instead of append for insertions
                signal_segments.extend(signals[counter:counter+length])
                mean_signal_segments.extend(mean_signals[counter:counter+length])
                counter += length
                ref_seq_with_deletions_insertions.extend([''] * length)
            elif op == "D":
                sequence.extend([''] * length)
                # Create a list of placeholder arrays (or values) for each deletion base
                signal_segments.extend([np.array([np.nan] * entry['stride'])] * length)
                mean_signal_segments.extend([np.nan] * length)
                ref_seq_with_deletions_insertions.extend(reference_sequence[ref_counter:ref_counter + length])
                ref_counter += length
            else:
                import warnings
                warnings.warn(f"Invalid CIGAR operator: '{op}'", UserWarning)
                continue
    
    
        result.append({
            'mapped_position_index': entry['mapped_position_index'],
            'sequence': np.array(sequence),
            'ref_seq': np.array(ref_seq_with_deletions_insertions, dtype='object'),
            'signal': np.array(signal_segments, dtype='object'),
            'mean_signal': np.array(mean_signal_segments, dtype='object')
        })

    return result
