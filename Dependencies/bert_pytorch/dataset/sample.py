from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split


def generate_pairs(line, window_size):
    line = np.array(line)
    line = line[:, 0]

    seqs = []
    for i in range(0, len(line), window_size):
        seq = line[i:i + window_size]
        seqs.append(seq)
    seqs += []
    seq_pairs = []
    for i in range(1, len(seqs)):
        seq_pairs.append([seqs[i - 1], seqs[i]])
    return seqs


def fixed_window(line, window_size, adaptive_window, seq_len=None, min_len=0, debug=False):
    if debug:
        print("- -"*20)
        print("Fixed Window on line:",line)

    line = [ln.split(",") for ln in line.split()] # 'key,time' mapping split
    if debug:
        print("splited line:",line)
        print("window_size:",window_size,"adaptive_window:",adaptive_window)
        print("seq_len:",seq_len,"min_len:",min_len)

    # filter the line/session shorter than 10
    if len(line) < min_len:
        if debug:
            print("len of line is lesser than min_len...\n\
            ...returning empty logkey_seqs & time_seqs")
        return [], []

    # max seq len
    if seq_len is not None:
        if debug:
            print("truncating line till seq_len")
        line = line[:seq_len]

    if adaptive_window:
        window_size = len(line)
        if debug:
            print("adaptive window size override:",window_size)

    line = np.array(line)
    if debug:
        print("line shape as np.array:",line.shape)

    # if time duration exists in data
    if line.shape[1] == 2:
        tim = line[:,1].astype(float)
        line = line[:, 0]
        if debug:
            print("Time Duration exists in Data")
            print("time:",tim,"\tline:",line)

        # the first time duration of a session should be 0, so max is window_size(mins) * 60
        tim[0] = 0
    else:
        line = line.squeeze()
        if (line.shape==()):
            line = np.array([line])
        # if time duration doesn't exist, then create a zero array for time
        tim = np.zeros(line.shape)
        
        if debug:
            print("Time Duration doesn't exists in Data (empty arr created)")
            print("time:",tim,"\tline:",line)

    logkey_seqs = []
    time_seq = []
    for i in range(0, len(line), window_size):
        logkey_seqs.append(line[i:i + window_size])
        time_seq.append(tim[i:i + window_size])

    return logkey_seqs, time_seq


def generate_train_valid(data_path, window_size=20, adaptive_window=True,
                         sample_ratio=1, valid_size=0.1, output_path=None,
                         scale=None, scale_path=None, seq_len=None, min_len=0,debug=False):
    with open(data_path, 'r') as f:
        data_iter = f.readlines()
    
    num_session = int(len(data_iter) * sample_ratio)
    # only even number of samples, or drop_last=True in DataLoader API
    # coz in parallel computing in CUDA, odd number of samples reports issue when merging the result
    # num_session += num_session % 2

    test_size = int(min(num_session, len(data_iter)) * valid_size)
    # only even number of samples
    # test_size += test_size % 2

    '''
    here n->num_session, x->len(data_iter), sr->sample_ratio, v->valid_size, t->test_size, vs=test_split_ratio
    n = x * sr
    t = min(n , x * v)
    => t = min(x * sr , x * v)
    => t = x * min(sr,v)
    ts = t/n
    => vs = [x * min(sr,v)] / [x * sr]
    => vs = [min(sr,v)] / [sr] 
    '''
    #split_size = round(test_size/num_session,3)
    # (or) to avoid precision loss by applying recall
    split_size = min(sample_ratio,valid_size)/sample_ratio
    # update split size

    print("before filtering short session")
    print("train size ", int(num_session - test_size))
    print("valid size ", int(test_size))
    
    if debug:
        print()
        print("= ="*20)
        print("Generate-Train-Valid on data_path:", data_path,"\n>>> Data:")
        print(data_iter)

    logkey_seq_pairs = []
    time_seq_pairs = []
    session = 0
    for line in tqdm(data_iter):
        if session >= num_session:
            break
        session += 1
        if debug:
            print("\t session:",session)
        logkeys, times = fixed_window(line, window_size, adaptive_window, seq_len, min_len, debug=debug)
        if debug:
            print()
            print("logkeys:",logkeys)
            print("times:",times)
        logkey_seq_pairs += logkeys
        time_seq_pairs += times

    logkey_seq_pairs = np.array(logkey_seq_pairs)
    time_seq_pairs = np.array(time_seq_pairs)
    if debug:
        print('- -'*20)
        print("\nlogkey_seq_pairs:",logkey_seq_pairs)
        print("\ntime_seq_pairs:",time_seq_pairs)
        print("\nvalidation split ratio:",split_size)
        print()

    logkey_trainset, logkey_validset, time_trainset, time_validset = train_test_split(logkey_seq_pairs,
                                                                                      time_seq_pairs,
                                                                                      test_size=split_size,
                                                                                      random_state=2023)

    # sort seq_pairs by seq len
    train_len = list(map(len, logkey_trainset))
    valid_len = list(map(len, logkey_validset))

    train_sort_index = np.argsort(-1 * np.array(train_len))
    valid_sort_index = np.argsort(-1 * np.array(valid_len))

    logkey_trainset = logkey_trainset[train_sort_index]
    logkey_validset = logkey_validset[valid_sort_index]

    time_trainset = time_trainset[train_sort_index]
    time_validset = time_validset[valid_sort_index]

    print("="*40)
    print("Num of train seqs", len(logkey_trainset))
    print("Num of valid seqs", len(logkey_validset))
    print("="*40)

    return logkey_trainset, logkey_validset, time_trainset, time_validset
