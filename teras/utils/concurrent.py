from multiprocessing import Pool
from tqdm import tqdm
from teras.training.listeners import ProgressBar
from more_itertools import chunked

def concurrent_apply(func, iterator, process_num=70):
    pool = Pool(processes=process_num)
    if not isinstance(iterator, list):
        iterator = list(iterator)
    total = len(iterator)
    pbar = ProgressBar(lambda n: tqdm(total=n))
    pbar.init(total)
    ret_list = []
    for ret in pool.imap_unordered(func, iterator):
        ret_list += [ret]
        pbar.update(1)
    pbar.close()
    return ret_list