import multiprocessing
import threading

import yaml
import os
import random

if 'APP_PATH' not in os.environ:
    raise Exception('APP_PATH environment variable must be provided')

# tool to generate random names
_alphabet_for_random_generation = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'


def _random_name(length=10):
    return ''.join([random.choice(_alphabet_for_random_generation) for i in range(length)])


_local = threading.local()
_local.total_slices = 1
_local.process_slice = 0

config = None


def init_config():
    global config
    config = None
    try:
        config_path = os.path.join(os.environ["APP_PATH"], 'config.yml')
        config_file = open(config_path, encoding='utf-8')
    except FileNotFoundError:
        raise Exception('A configuration file "config.yml" must be placed at the root of the application directory.')

    try:
        config = yaml.safe_load(config_file)
    except Exception as e:
        raise Exception('Failed to load configuration file : ' + str(e))

    config["app_path"] = os.environ["APP_PATH"]
    config["subprocess"] = multiprocessing.current_process().name != "MainProcess"

    if 'dataset_path' not in config:
        config["dataset_path"] = os.path.join(config['app_path'], "data/dataset")

    def _get_machine_id():
        mid_path = os.path.join(config['dataset_path'], "_machine_id")
        if os.path.isfile(mid_path):
            with open(mid_path, 'r') as f:
                return f.read()
        else:
            mid = _random_name(5)
            os.makedirs(os.path.split(mid_path)[0], exist_ok=True)
            with open(mid_path, 'w') as f:
                f.write(mid)
            return mid

    config['machine_id'] = _get_machine_id()
    config["pbar"] = {
        'input': False,
        'output': False
    }


init_config()
