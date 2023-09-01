import re

from lib.config import config
from lib.mongodb import mongo_connection

from concurrent.futures import ThreadPoolExecutor
import json
from tqdm import tqdm

# Associates function_id with a service planning
user_functions = {}

# Associates function_id with a progress bar configuration
pbar_config = {}


def Scraper(name, dataset=None):
    """
    Register service as scrapper.
    """

    def transformer(service_planning):
        if not isinstance(service_planning, tuple):
            service_planning = ('[]', service_planning)

        # in a subprocess do nothing
        if config['subprocess']:
            return service_planning[1]

        if not dataset and not name:
            raise Exception('At least a step name or a dataset name must be provided for a Scrapper.')

        global user_functions

        # load decorators as (name, (*args)) list
        decorators = json.loads(service_planning[0])

        # compute a unique identifier of the procedure to use as a marker in the db
        function_id = '_scrap_' + name

        # Default progress bar configuration is None
        pbar_config[function_id] = set()

        # Add dataset_name to the arguments of Input and Output
        # it allows them to store this value in the mongo documents
        # also specify to a potential Processes decorator if it
        # needs to retrieve the output data of its subprocesses
        need_output = False
        need_input = False
        for dec in decorators:
            if dec[0] == "Output":
                need_output = True
                dec[2]['function_id'] = function_id
                if dataset:
                    dec[2]['dataset_name'] = dataset

                # Check for a progress bar
                if 'pbar' in dec[2]:
                    if dec[2]['pbar'] is True:
                        pbar_config[function_id].add('output')
                    del dec[2]['pbar']
            elif dec[0] == "Input":
                need_input = dec[1][0] not in ('feed', 'pipe')
                dec[2]['function_id'] = function_id
                if dataset:
                    dec[2]['dataset_name'] = dataset

                # Check for a progress bar
                if 'pbar' in dec[2]:
                    if dec[2]['pbar'] is True:
                        pbar_config[function_id].add('input')
                    del dec[2]['pbar']
            elif dec[0] == "Processes":
                if need_output:
                    dec[1].append(True)

        # Add a clerk to mark items in and out
        if need_input:
            decorators = (*decorators, ('Clerk', [function_id], {}))

        # initialize if needed before launching the pipeline and potentially forking
        service_planning = (json.dumps(decorators), service_planning[1])

        # store the pipeline under the appropriate command
        user_functions[function_id] = service_planning

        return service_planning[1]
    return transformer



def Preprocessor(name):
    """
    Register service as Preprocessor.
    """

    def transformer(service_planning):
        if not isinstance(service_planning, tuple):
            service_planning = ('[]', service_planning)

        # in a subprocess do nothing
        if config['subprocess']:
            return service_planning[1]

        global user_functions

        # load decorators as (name, (*args)) list
        decorators = json.loads(service_planning[0])

        # compute a unique identifier of the procedure to use as a marker in the db
        function_id = '_preprocess_' + name

        # Default progress bar configuration is None
        pbar_config[function_id] = set()

        # Specify to a potential Processes decorator if it
        # needs to retrieve the output data of its subprocesses
        need_output = False
        need_input = False
        for dec in decorators:
            if dec[0] == "Output":
                need_output = True
                dec[2]['function_id'] = function_id

                # Check for a progress bar
                if 'pbar' in dec[2]:
                    if dec[2]['pbar'] is True:
                        pbar_config[function_id].add('output')
                    del dec[2]['pbar']
            elif dec[0] == "Input":
                need_input = dec[1][0] not in ('feed', 'pipe')
                dec[2]['function_id'] = function_id

                # Check for a progress bar
                if 'pbar' in dec[2]:
                    if dec[2]['pbar'] is True:
                        pbar_config[function_id].add('input')
                    del dec[2]['pbar']
            elif dec[0] == "Processes":
                if need_output:
                    dec[1].append(True)

        # Add a clerk to mark items in and out
        if need_input:
            decorators = (*decorators, ('Clerk', [function_id], {}))

        service_planning = (json.dumps(decorators), service_planning[1])

        # Store the pipeline under the appropriate command
        user_functions[function_id] = service_planning

        return service_planning[1]

    return transformer

def Trainer(name):
    """
    Register service as Trainer.
    """

    def transformer(service_planning):
        if not isinstance(service_planning, tuple):
            service_planning = ('[]', service_planning)

        # in a subprocess do nothing
        if config['subprocess']:
            return service_planning[1]

        global user_functions

        # load decorators as (name, (*args)) list
        decorators = json.loads(service_planning[0])

        # compute a unique identifier of the procedure to use as a marker in the db
        function_id = '_train_' + name

        # specify to a potential Processes decorator if it
        # needs to retrieve the output data of its subprocesses
        need_input = False
        has_load = False
        for dec in decorators:
            if dec[0] == "Output":
                need_input = True
                dec[2]['function_id'] = function_id
            elif dec[0] == "Input":
                dec[2]['function_id'] = function_id
            elif dec[0] == "Processes":
                if need_input:
                    dec[1].append(True)
            elif dec[0] == "Load":
                has_load = True

        # Check for a @Load decorator
        if not has_load:
            raise Exception('@Load mandatory for a Trainer')

        # Add train decorator
        decorators = (*decorators, ("Train", [], {}))
        service_planning = (json.dumps(decorators), service_planning[1])

        # Store the pipeline under the appropriate command
        user_functions[function_id] = service_planning

        return service_planning[1]
    return transformer


def Importer(name, dataset, query=None, exclude=None):
    # make exclude into a projection
    projection = {}
    for ex in exclude:
        projection[ex] = 0

    # set query
    if not query:
        query = {}

    def transformer(func):
        # if a decorator was used, ignore it
        if isinstance(func, tuple):
            func = func[1]

        def wrapper():
            compartment_list = set()

            main_it = mongo_connection[f'{dataset}-dataset'].find(query, projection)
            s3_it = mongo_connection[f'{dataset}-dataset-s3'].find({})

            with ThreadPoolExecutor(max_workers=250) as workers:
                # pass regular documents through the user function
                pbar = tqdm(func(main_it))
                pbar.set_description_str(f'Importing {dataset} documents...')
                for item in pbar:
                    # delete useless file fields
                    for prop in item:
                        # delete _tar_* fields
                        rm = re.match('_tar_(.*)', prop)
                        if rm:
                            if rm[1] not in item:
                                del item[prop]
                            else:
                                compartment_list.add(rm[1])
                        # delete _local_* fields
                        else:
                            rm = re.match('_local_(.*)', prop)
                            if rm:
                                if rm[1] not in item:
                                    del item[prop]
                                else:
                                    compartment_list.add(rm[1])

                    workers.submit(mongo_connection[config['project_name'] + '-pipeline'].insert_one, item)

                # transfer all s3 documents
                pbar = tqdm(s3_it)
                pbar.set_description_str(f'Importing {dataset} s3 documents...')
                for item in pbar:
                    # check if the batch corresponds to any document in the main collection
                    if item['field'] not in compartment_list:
                        continue
                    workers.submit(mongo_connection[config['project_name'] + '-pipeline-s3'].insert_one, item)

        # register the service
        user_functions[name] = wrapper

    return transformer
