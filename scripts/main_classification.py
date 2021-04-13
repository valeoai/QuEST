from __future__ import print_function

import os

from sacred import Experiment, SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import MongoObserver

from distillation.algorithms.classification.classification import Classification
from distillation.algorithms.classification.classification_quest import ClassificationQUEST
from distillation.dataloaders.basic_dataloaders import SimpleDataloader
from distillation.dataloaders.basic_dataloaders import DataloaderSampler
from distillation.datasets import dataset_factory
from distillation import project_root

SETTINGS.CAPTURE_MODE = 'sys'

ex = Experiment('classification_distillation')


ex.captured_out_filter = apply_backspaces_and_linefeeds

command_line_arguments = dict(config='',
                              checkpoint=0,
                              num_workers=4,
                              cuda=True,
                              disp_step=200,
                              localpath='',
                              evaluate=False,
                              testset=False,
                              best=False,
                              kmeans=0,
                              fname=None,
                              ksubset=None,
                              data_dir=None,
                              memmap=False,
                              memmap_dir='./tmp')
ex.add_config(command_line_arguments)

@ex.automain
def main_experiment(config,
                    checkpoint,
                    num_workers,
                    cuda,
                    disp_step,
                    localpath,
                    evaluate,
                    testset,
                    best,
                    kmeans,
                    fname,
                    ksubset,
                    data_dir,
                    memmap,
                    memmap_dir,
                    _run,
                    _log):
    exp_config_file = config
    exp_base_directory = (
        os.path.join(project_root, 'experiments') if localpath=='' else localpath)
    if kmeans > 0:
        exp_base_directory = os.path.join(exp_base_directory, 'VQ')

    exp_directory = os.path.join(exp_base_directory, config.replace('.', '/'))

    # Load the configuration params of the experiment
    print('Launching experiment: %s' % exp_config_file)
    config = __import__(exp_config_file, fromlist=['']).config

    _run.meta_info['container_name'] = os.environ.get('CONTAINER_NAME', 'no name')
    _run.meta_info['config_loaded'] = config

    config['exp_dir'] = exp_directory

    print(f"Loading experiment {config} from file: {exp_config_file}")
    print(f"Generated logs, snapshots, and model files will be stored on {config['exp_dir']}")

    # Set train and test datasets and the corresponding data loaders
    data_train_opt = config['data_train_opt']
    data_test_opt = config['data_test_opt']
    dataset_train_args = data_train_opt['dataset_args']
    dataset_test_args = data_test_opt['dataset_args']
    if data_dir is not None:
        dataset_train_args["data_dir"] = data_dir
        dataset_test_args["data_dir"] = data_dir
    batch_size_test = data_test_opt.get('batch_size', data_train_opt['batch_size'])

    if testset:
        dataset_test_args['split'] = 'test'

    dataset_test = dataset_factory(
        dataset_name=data_test_opt['dataset_name'], **dataset_test_args)
    dloader_test = SimpleDataloader(
        dataset=dataset_test,
        batch_size=batch_size_test,
        num_workers=num_workers,
        train=False)

    config['disp_step'] = disp_step

    algorithm_type = config['algorithm_type']
    if (algorithm_type == 'classification.classification_quest' or kmeans > 0):
        algorithm = ClassificationQUEST(config, _run, _log)
    elif algorithm_type == 'classification.classification':
        algorithm = Classification(config, _run, _log)

    if cuda:  # enable cuda
        algorithm.load_to_gpu()

    if evaluate:
        suffix = '.best' if best else ''
        algorithm.load_checkpoint(epoch='*', train=False, suffix=suffix)
        algorithm.evaluate(dloader_test, use_tensorboard=False)

    elif kmeans > 0:
        algorithm.load_checkpoint(epoch='*', train=False, suffix='')
        dataset_train_args['do_not_use_random_transf'] = True
        dataset_train = dataset_factory(
            dataset_name=data_train_opt['dataset_name'], **dataset_train_args)

        if ksubset is None:
            dloader_train = SimpleDataloader(
                dataset=dataset_train,
                batch_size=data_train_opt['batch_size'],
                num_workers=num_workers,
                epoch_size=None,
                train=False)
        else:
            assert isinstance(ksubset, int)
            dloader_train = SimpleDataloader(
                dataset=dataset_train,
                batch_size=data_train_opt['batch_size'],
                num_workers=num_workers,
                epoch_size=ksubset,
                train=True)


        algorithm.apply_kmeans_to_dataset(
            dataloader=dloader_train,
            num_embeddings=kmeans,
            feature_name=fname,
            memmap=memmap,
            memmap_dir=memmap_dir)
    else:
        if checkpoint != 0:  # load checkpoint
            algorithm.load_checkpoint(
                epoch=checkpoint if (checkpoint > 0) else '*', train=True)

        dataset_train = dataset_factory(
            dataset_name=data_train_opt['dataset_name'], **dataset_train_args)

        use_ds_dataloader = config.get('use_ds_dataloader', False)
        if use_ds_dataloader:
            dloader_train = DataloaderSampler(
                dataset=dataset_train,
                batch_size=data_train_opt['batch_size'],
                num_workers=num_workers,
                train=True)
        else:
            dloader_train = SimpleDataloader(
                dataset=dataset_train,
                batch_size=data_train_opt['batch_size'],
                num_workers=num_workers,
                epoch_size=data_train_opt['epoch_size'],
                train=True)

        # train the algorithm
        algorithm.solve(dloader_train, dloader_test)
