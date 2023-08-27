import decimal
import multiprocessing
import os
import copy
import types

from lib.iterable import _generator
from lib.mongodb import mongo_connection
from lib.config import config
from lib.s3 import _iterate_batch_s3, _get_image

from torch.utils.data import IterableDataset, DataLoader, Dataset, random_split
import torchvision.transforms
import torch
import random
from collections import Counter
from PIL import Image

from loguru import logger


class MongoDataset(Dataset):
    def __init__(self, data_field, target_field, preprocess, composition, query):
        collection = mongo_connection[config['project_name'] + "-pipeline"]

        query[data_field] = {'$exists': True}
        selection = {
            data_field: 1
        }
        if target_field:
            query[target_field] = {'$exists': True}
            selection[target_field] = 1

        self.item_list = []

        if preprocess:
            preprocess = preprocess()

        for dataset, quantity in composition.items():
            special_query = copy.deepcopy(query)
            special_query['dataset'] = dataset
            cursor = collection.find(special_query, selection, batch_size=quantity).limit(quantity)
            for item in cursor:
                data = item[data_field]
                if target_field in item:
                    target = item[target_field]
                    self.item_list.append(
                        preprocess(data, target) if preprocess else (data, target))
                else:
                    self.item_list.append(
                        preprocess(data) if preprocess else data)

    def __getitem__(self, index):
        return self.item_list[index]

    def __len__(self):
        return len(self.item_list)


class LocalDataset(Dataset):
    def __init__(self,
                 data_field,
                 target_field,
                 preprocess,
                 composition,
                 query=None):
        self.data_field = data_field
        self.target_field = target_field
        self.data_list = []

        machine_id = config["machine_id"]
        query[f'_local/{machine_id}/{data_field}'] = True
        if target_field:
            query[target_field] = {'$exists': True}

        # get only dataset, data field, target field
        projection = {
            'dataset': True,
            data_field: True
        }
        if target_field:
            projection[target_field] = True

        # get all documents
        self.all_docs = []
        for dataset, count in composition.items():
            sub_query = query.copy()
            sub_query['dataset'] = dataset
            self.all_docs.extend(mongo_connection[f'{config["project_name"]}-pipeline'].find(
                query,
                projection
            ).limit(count))

        # shuffle
        random.shuffle(self.all_docs)

        self.preprocess = preprocess
        self.initialized = False

    def __getitem__(self, item):
        # check if the preprocess function has been initialized
        if not self.initialized:
            if self.preprocess:
                self.preprocess = self.preprocess()
            self.initialized = True

        doc = self.data_list[item]
        file_name = doc[self.data_field]
        dataset = doc['dataset']

        data = Image.open(f'{config["dataset_path"]}/{dataset}/{self.data_field}/{file_name}')

        if self.target_field:
            if self.preprocess:
                return self.preprocess(data, doc[self.target_field])
            else:
                return data, doc[self.target_field]
        else:
            if self.preprocess:
                return self.preprocess(data)
            else:
                return data

    def __len__(self):
        return len(self.data_list)


class S3PickDataset(Dataset):
    def __init__(self,
                 field,
                 target_field,
                 composition,
                 query,
                 preprocess):
        self.field = field
        self.target_field = target_field

        query[f'_tar_{field}'] = {'$exists': True}
        query[f'_pos_{field}'] = {'$exists': True}

        projection = {
            f'_tar_{field}': True,
            f'_pos_{field}': True,
            'dataset': True
        }

        if target_field:
            query[target_field] = {'$exists': True}
            projection[target_field] = True

        self.all_docs = []
        for dataset, count in composition.items():
            query = query.copy()
            query['dataset'] = dataset
            self.all_docs.extend(mongo_connection[f'{config["project_name"]}-pipeline'].find(
                query,
                projection
            ).limit(count))
        random.shuffle(self.all_docs)

        self.initialized = False
        self.preprocess = preprocess

    def __getitem__(self, item):
        if not self.initialized:
            if self.preprocess:
                self.preprocess = self.preprocess()
            self.initialized = True

        doc = self.all_docs[item]

        if self.target_field:
            if self.preprocess:
                return self.preprocess(_get_image(doc, self.field), doc[self.target_field])
            else:
                return _get_image(doc, self.field), doc[self.target_field]
        else:
            if self.preprocess:
                return self.preprocess(_get_image(doc, self.field))
            else:
                return _get_image(doc, self.field)

    def __len__(self):
        return len(self.all_docs)


class SliceShuffleBuffer:
    def __init__(self, generator, shuffle_size, sh_limit, slice_size=None):
        self.shuffle_size = shuffle_size
        self.slice_size = slice_size

        self.it = iter(generator)
        self.pos = 0

        self.current_slice_size = 0
        self.start_byte = 0
        self.end_of_slice = False
        self.slice_bounds = None

        self.buffer = []
        self.full_stop = False

        self.sh = len(sh_limit) > 2
        if self.sh:
            self.sh_lock, self.sh_dict, self.sh_key = sh_limit
        else:
            self.sh_dict, self.sh_key = sh_limit

    def sh_inc(self):
        if self.sh:
            with self.sh_lock:
                self.sh_dict[self.sh_key] -= 1
        else:
            self.sh_dict[self.sh_key] -= 1

    def __next__(self):
        while (not self.full_stop and
               len(self.buffer) < self.shuffle_size):
            # stop filling the buffer at the end of the slice
            if self.slice_size and self.current_slice_size > self.slice_size:
                break

            # if no more shared credit
            if self.sh_dict[self.sh_key] <= 0:
                self.full_stop = True
                break

            self.sh_inc()

            # get next item
            try:
                doc, img, (_, self.pos) = next(self.it)
                self.buffer.append((doc, img))
            except StopIteration:
                self.full_stop = True

        # full end of buffer
        if len(self.buffer) == 0:
            raise StopIteration

        # at the end of the buffer, register the slice
        if self.slice_size and len(self.buffer) == 1:
            self.slice_bounds = (self.start_byte, self.pos)
            self.end_of_slice = True

            self.current_slice_size = 0
            self.start_byte = self.pos

        return self.buffer.pop(random.randint(0, len(self.buffer)-1))

    def get_slice(self):
        if self.end_of_slice:
            self.end_of_slice = False
            return self.slice_bounds
        return None



class S3Dataset(IterableDataset):
    def __init__(self,
                 batch_list,  # list of batches as mongo documents
                 target_field,  # name of the field for the target
                 num_workers,  # number of workers of the dataloader
                 preprocess,  # () -> ((img, target) -> (img, target))
                 composition,  # dict: dataset -> max number of elements
                 slice_size=100,  # size of slices
                 shuffle_size=50,  # size of shuffle buffer given to s3 iterator
                 query=None):
        super(S3Dataset).__init__()

        self.num_workers = num_workers

        self.composition = composition

        self.progress = None
        if self.composition:
            if self.num_workers > 1:
                manager = multiprocessing.Manager()
                self.progress = manager.dict()
                for key in composition:
                    self.progress[key] = composition[key]
                self.progress_lock = multiprocessing.Lock()
            else:
                self.progress = {}

        self.generator = None
        self.batch_list = [[] for i in range(num_workers)]

        random.shuffle(batch_list)
        for gen_index, batch_doc in enumerate(batch_list):
            self.batch_list[gen_index % num_workers].append(batch_doc)

        self.target_field = target_field

        self.preprocess = preprocess

        self.initialized = False
        self.slices = {}
        self.slice_size = slice_size
        self.shuffle_size = shuffle_size

        self.dataset_list = None
        self.worker_id = None

        self.query = query

    def initialize_worker(self):
        if not self.preprocess:
            to_tensor = torchvision.transforms.PILToTensor()

            def preprocess(img, target):
                return to_tensor(img), target

            self.preprocess = preprocess
        else:
            self.preprocess = self.preprocess()
            self.preprocess_init = True

    def get_slice(self, slice_doc):
        kwargs = {}
        if self.target_field:
            kwargs['projection'] = {
                self.target_field: 1
            }

        if self.num_workers > 1:
            sh_limit = (self.progress_lock, self.progress, slice_doc['dataset'])
        else:
            sh_limit = (self.progress, slice_doc['dataset'])

        if 'bounds' in slice_doc:
            return SliceShuffleBuffer(_iterate_batch_s3(
                    slice_doc,
                    get_bounds=True,
                    slice=slice_doc['bounds'],
                    query=self.query,
                    **kwargs
                ),
                self.shuffle_size,
                sh_limit)
        else:
            return SliceShuffleBuffer(_iterate_batch_s3(
                    slice_doc,
                    get_bounds=True,
                    query=self.query,
                    **kwargs
                ),
                self.shuffle_size,
                sh_limit,
                self.slice_size)

    def __iter__(self):
        if self.composition:
            for key, value in self.composition.items():
                self.progress[key] = value

        worker_info = torch.utils.data.get_worker_info()
        self.worker_id = 0 if worker_info is None else worker_info.id

        up_count = 0
        dataset_up = {}

        # ********* For the first pass *********
        first_pass = not self.initialized
        if not self.initialized:
            self.initialize_worker()
            self.initialized = True

            # if not parallelized
            if worker_info is None:
                assert self.num_workers == 1
            # if parallelized
            else:
                assert self.num_workers == worker_info.num_workers

            count = Counter([batch_doc["dataset"] for batch_doc in self.batch_list[self.worker_id]])
            self.dataset_list, nb_per_dataset = list(count.keys()), list(count.values())

            logger.info(f'Datasets for worker {self.worker_id} : {self.dataset_list} {nb_per_dataset}')

            # First slices are just batches
            self.slices = {}
            for name in self.dataset_list:
                # Select only the batches of this dataset
                self.slices[name] = ([batch_doc for batch_doc in self.batch_list[self.worker_id] if
                                      batch_doc["dataset"] == name])

            builded_slices = {dataset_name: [] for dataset_name in self.dataset_list}
        # *********

        # Shuffle slices
        for dataset_name in self.dataset_list:
            random.shuffle(self.slices[dataset_name])

        slices = copy.deepcopy(self.slices)

        # Initialize slice_doc and batch_it with the first slice of every dataset
        current_slice_doc = {dataset_name: slices[dataset_name].pop() for dataset_name in
                             self.dataset_list}
        current_slice_it = {dataset_name: self.get_slice(current_slice_doc[dataset_name]) for dataset_name in
                            self.dataset_list}

        # reset loop condition
        for dataset_name in self.dataset_list:
            up_count += 1
            dataset_up[dataset_name] = True

        while up_count > 0:
            # Pick a dataset at random
            total_images = sum(self.composition.values())
            if total_images == 0:
                n = len(self.composition)
                probabilities = {dataset: 1 / n for dataset in self.composition.keys()}
            else:
                probabilities = {dataset: count / total_images for dataset, count in self.composition.items()}
            dataset_name = random.choices(list(probabilities.keys()), weights=list(probabilities.values()), k=1)[0]
            if dataset_name not in self.dataset_list:
                continue

            # Check if the dataset is still up
            if not dataset_up[dataset_name]:
                continue

            # Get the item
            try:
                doc, img = next(current_slice_it[dataset_name])
            except StopIteration:
                # Slice the last part of the batch
                if first_pass:
                    build_slice = current_slice_doc[dataset_name].copy()
                    build_slice['bounds'] = current_slice_it[dataset_name].get_slice()
                    builded_slices[dataset_name].append(build_slice)

                # end of iteration for dataset
                if self.progress[dataset_name] <= 0 or len(slices[dataset_name]) == 0:
                    dataset_up[dataset_name] = False
                    up_count -= 1
                    continue

                # next batch
                current_slice_doc[dataset_name] = slices[dataset_name].pop()
                current_slice_it[dataset_name] = self.get_slice(current_slice_doc[dataset_name])

                # get the first item
                doc, img = next(current_slice_it[dataset_name])
            else:
                # check for a slice
                if first_pass:
                    mb_bounds = current_slice_it[dataset_name].get_slice()
                    if mb_bounds:
                        build_slice = current_slice_doc[dataset_name].copy()
                        build_slice['bounds'] = mb_bounds
                        builded_slices[dataset_name].append(build_slice)


            # Build the item, preprocess it, check the result
            if self.target_field:
                if self.target_field not in doc:
                    continue

                if self.preprocess:
                    item = self.preprocess(img, doc[self.target_field])
                    if not item:
                        continue
                else:
                    item = img, doc[self.target_field]
            else:
                if self.preprocess:
                    item = self.preprocess(img)
                else:
                    item = img

            yield item

        if first_pass:
            self.slices = builded_slices


def get_dataloaders(data_field, target_field, ratios, batch_size, num_workers, preprocess,
                    composition, source, pick, query, shuffle_size, slice_size):
    collection = mongo_connection[config['project_name'] + "-pipeline"]
    collection_s3 = mongo_connection[config['project_name'] + "-pipeline-s3"]

    if source == "s3":
        if not pick:
            # get all batch documents from MongoDB
            batch_list = list(collection_s3.find({'field': data_field}))

            if ratios:
                # split the batches randomly
                random.Random(config['train_seed']).shuffle(batch_list)
                slices = [int(len(batch_list) * s) for s in ratios]
                tvt_batches = (batch_list[0:slices[0]],
                               batch_list[slices[0]:slices[1] + slices[0]],
                               batch_list[slices[1] + slices[0]:])

                # compute compositions for each subset
                compositions = [{key: value * ratios[i] for key, value in composition.items()} for i in range(len(ratios))]

                # build datasets
                tvt_datasets = (S3Dataset(sub_batch_list,
                                          target_field,
                                          num_workers,
                                          preprocess,
                                          composition,
                                          query=query,
                                          shuffle_size=shuffle_size,
                                          slice_size=slice_size) for
                                    sub_batch_list, composition in
                                    zip(tvt_batches, compositions))
            else:
                dataset = S3Dataset(batch_list, target_field, num_workers, preprocess, composition,
                                    shuffle_size=shuffle_size,
                                    slice_size=slice_size,
                                    query=query)
        else:
            dataset = S3PickDataset(
                data_field,
                target_field,
                composition,
                query,
                preprocess
            )
            if ratios:
                tvt_datasets = random_split(
                    dataset,
                    ratios,
                    generator=torch.Generator().manual_seed(config['train_seed'])
                )

    elif source == "local":
        dataset = LocalDataset(
            data_field,
            target_field,
            preprocess,
            composition,
            query
        )
        if ratios:
            tvt_datasets = random_split(
                dataset,
                ratios,
                generator=torch.Generator().manual_seed(config['train_seed'])
            )
    else:
        dataset = MongoDataset(data_field, target_field, preprocess, composition, query)
        if ratios:
            tvt_datasets = random_split(
                dataset,
                ratios,
                generator=torch.Generator().manual_seed(config['train_seed'])
            )

    if ratios:
        tvt_dataloaders = [DataLoader(
            subset,
            batch_size=batch_size,
            generator=torch.Generator().manual_seed(config['train_seed']),
            num_workers=num_workers,
            persistent_workers=True,
            pin_memory=True,
            shuffle=(source != 's3' or pick),
            drop_last=True
        ) for subset in tvt_datasets]

        return tvt_dataloaders
    else:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            generator=torch.Generator().manual_seed(config['train_seed']),
            num_workers=num_workers,
            persistent_workers=True,
            pin_memory=True,
            shuffle=(source != 's3' or pick),
            drop_last=True
        )


def _Load(data,
          batch_size,
          composition=None,
          split=None,
          target=None,
          num_workers=os.cpu_count() - 1,
          preprocess=False,
          source=None,
          query=None,
          pick=False,
          shuffle_size=1000,
          slice_size=50
          ):
    def transformer(func, sub_proc_wrapped):
        def wrapper(*args, **kwargs):
            nonlocal query
            nonlocal pick

            logger.info(f"Loading data from field '{data}'")

            preprocess_fn = None
            if preprocess:
                preprocess_fn = sub_proc_wrapped[1].__preprocess__

            if not query:
                query = {}

            tvt_dataloaders = get_dataloaders(data,
                                              target,
                                              split,
                                              batch_size,
                                              num_workers,
                                              preprocess_fn,
                                              composition,
                                              source,
                                              pick,
                                              query,
                                              shuffle_size,
                                              slice_size)

            for item in _generator(func, tvt_dataloaders):
                yield item

        return wrapper

    return transformer
