import os
import yaml
from multiprocessing import Pool
from functools import partial
import subprocess

BASE_CONFIG_FILE = 'experiments/cfgs/base_config.yml'
PARALLEL_EXPERIMENTS = 2
ALL_EXPERIMENTS = 'experiments/experiments_list.txt'
COMPLETED_EXPERIMENTS = 'experiments/completed_experiments.txt'
TEMPORARY_CONFIGS = 'temp'
if not os.path.exists(TEMPORARY_CONFIGS):
    os.makedirs(TEMPORARY_CONFIGS)

def rec_merge(main_dict, mixin_dict):

    for key in mixin_dict.keys():
        if isinstance(main_dict[key], dict):
            merge_dict = rec_merge(
                main_dict[key], mixin_dict[key])
            main_dict[key] = merge_dict
        else:
            main_dict[key] = mixin_dict[key]
    return main_dict


def run_experiment(base_cfg, exp_mixin):

    mixed_config = rec_merge(base_cfg, exp_mixin)
    config_name = 'temp_config_proc_{0}.yml'.format(os.getpid())
    temp_config_path = os.path.join(
        TEMPORARY_CONFIGS, config_name)
    with open(temp_config_path, 'w') as f:
        yaml.dump(mixed_config, f, default_flow_style=False)

    subprocess.call(['python3', './tools/train_net.py',
        '--cfg', temp_config_path])
    with open(COMPLETED_EXPERIMENTS, 'a') as f:
        f.write(repr(exp_mixin) + '\n')

if __name__ == '__main__':

    print('Load base config')
    with open(BASE_CONFIG_FILE, 'r') as f:
        base_cfg = yaml.load(f)
    with open(ALL_EXPERIMENTS, 'r') as f:
        experiments_list = f.readlines()
    with open(COMPLETED_EXPERIMENTS, 'r') as f:
        completed_list = f.readlines()

    # Convert to dicts
    experiments_list = [eval(exp) for exp in experiments_list]
    completed_list = [eval(exp) for exp in completed_list]

    exp_to_process = [exp for exp in experiments_list if exp not in completed_list]
    print(exp_to_process)

    pool = Pool(processes=PARALLEL_EXPERIMENTS)

    func = partial(run_experiment, base_cfg)
    pool.map(func, exp_to_process)
