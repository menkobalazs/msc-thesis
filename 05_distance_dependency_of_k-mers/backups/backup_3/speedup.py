import numpy as np
from tqdm import tqdm
from ont_fast5_api.fast5_interface import get_fast5_file
import warnings

def read_fast5(fname, single_or_multi_read, get_read_ids=False):
    'read only one fast5 file'
    from ont_fast5_api.fast5_interface import get_fast5_file

    def raw_to_current(rawdata, offset, range_, digitisation):
        return (rawdata + offset) * range_ / digitisation  # Keep as NumPy array

    if single_or_multi_read not in ['s', 'm']:
        raise ValueError('single_or_multi variable must be "s" or "m".')

    data = []
    read_ids = []
    with get_fast5_file(fname, mode="r") as f:
        for read in f.get_reads():
            ch = read.get_channel_info()
            data.append(raw_to_current(read.get_raw_data(), ch['offset'], ch['range'], ch['digitisation']))
            read_ids.append(read.read_id)

    data = np.array(data, dtype='object')  # Use NumPy object array to handle variable-length signals
    return (data[0], read_ids) if single_or_multi_read == 's' else (data, read_ids)



def find_read_id_index(read_id_list, searched_read_id):
    index = np.flatnonzero(np.char.find(read_id_list, searched_read_id) != -1)
    return index[0] if index.size == 1 else None


def split_raw_signal(raw_signal, move_table, stride):
    start_of_bases = np.where(move_table == 1)[0] * stride
    if len(start_of_bases) < 2:
        return start_of_bases, [], []

    signals = [raw_signal[i:j] for i, j in zip(start_of_bases, start_of_bases[1:])]
    mean_signals = np.array([np.mean(raw_signal[i:j]) for i, j in zip(start_of_bases, start_of_bases[1:])])
    return start_of_bases, signals, mean_signals


def get_raw_signal_with_bases(sam_data, fast5_path, reference_sequence, verbose=False):
    import warnings
    result = []

    for entry in tqdm(sam_data, total=len(sam_data), desc='Processing data'):
        fast5_signals, read_ids = read_fast5(fast5_path + entry['fast5_file_name'], 'm', get_read_ids=True)

        read_index = find_read_id_index(np.array(read_ids), entry['read_id'])
        if read_index is None:
            warnings.warn(f"Read ID {entry['read_id']} not found in {entry['fast5_file_name']}. Skipping.", UserWarning)
            continue

        trimmed_signal = fast5_signals[read_index][entry['trim_offset']:]
        _, signals, _ = split_raw_signal(trimmed_signal, entry['move_table'], entry['stride'])

        sequence = np.empty(len(entry['sequence']), dtype='U1')
        ref_seq_with_deletions_insertions = np.empty(len(reference_sequence), dtype='U1')
        signal = np.empty(len(entry['sequence']), dtype='object')

        counter = 0
        ref_counter = entry['mapped_position_index']

        for op, length in zip(entry['cigar_str'][1], entry['cigar_str'][0]): 
            if op == "S":  
                counter += length  

            elif op == "M":  
                sequence[counter:counter + length] = entry['sequence'][counter:counter + length]  
                signal[counter:counter + length] = signals[counter]
                ref_seq_with_deletions_insertions[ref_counter:ref_counter + length] = reference_sequence[ref_counter:ref_counter + length] 
                ref_counter += length  
                counter += length  

            elif op == "I":  
                sequence[counter:counter + length] = entry['sequence'][counter:counter + length]  
                signal[counter:counter + length] = signals[counter]
                ref_seq_with_deletions_insertions[ref_counter:ref_counter + length] = ''  
                counter += length  

            elif op == "D":  
                sequence[counter:counter + length] = ''  
                signal[counter:counter + length] = None  
                ref_seq_with_deletions_insertions[ref_counter:ref_counter + length] = reference_sequence[ref_counter:ref_counter + length]  
                ref_counter += length  

            else:  
                warnings.warn(f"Invalid CIGAR operator: '{op}'", UserWarning)  
                break  

        result.append({
            'sequence': sequence,
            'ref_seq': ref_seq_with_deletions_insertions,
            'signal': signal
        })

    return result


