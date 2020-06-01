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


import subprocess
import json

DEFAULT_ATTRIBUTES = (
    'index',
    'uuid',
    'name',
    'timestamp',
    'memory.total',
    'memory.free',
    'memory.used',
    'utilization.gpu',
    'utilization.memory'
)

def get_gpu_info(nvidia_smi_path='nvidia-smi', keys=DEFAULT_ATTRIBUTES, no_units=True):
    nu_opt = '' if not no_units else ',nounits'
    cmd = '%s --query-gpu=%s --format=csv,noheader%s' % (nvidia_smi_path, ','.join(keys), nu_opt)
    output = subprocess.check_output(cmd, shell=True)
    lines = output.decode().split('\n')
    lines = [ line.strip() for line in lines if line.strip() != '' ]

    return [ { k: v for k, v in zip(keys, line.split(', ')) } for line in lines ]