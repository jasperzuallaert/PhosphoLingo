import json
import os.path

import pytorch_lightning as pl
from datetime import datetime

from lightning_module import LightningModule
from pytorch_lightning.callbacks import EarlyStopping
import input_reader as ir
import utils
from torch.utils.data.dataloader import DataLoader
from input_tokenizers import tokenizers


def run_training(json_file = None):
    """
    Main function to run training, and to evaluate and save the resulting model

    Parameters
    ----------
    json_file : str
        file location of the configuration file. See ``configs/default_config.json`` for an example of the fields to be
        filled in. If not all fields are specified, entries in default_config.json will be set as default values.

    """
    json_name = os.path.basename(json_file[:-5])
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # (1) parse .json config file
    config = json.loads(utils.DEFAULT_JSON)
    for key, value in json.load(open(json_file)).items():
        if key in config:
            config[key] = value
        else:
            print(f'Warning: provided option {key} not recognized -> discarded')
    config['json_file'] = json_file

    assert config['conv_depth'] * config['max_pool_size'] <= config['receptive_field'], \
        'The receptive field should be large enough to accommodate the specified pooling size'

    # (2) Set up logging
    logger = pl.loggers.CSVLogger(save_dir=f'logs', name=json_name, version=timestamp)
    logger.log_hyperparams(config)

    # (3) Set the batch size to run on the GPU, as well as the gradient accumulation parameter in case this exceeds
    #     the maximum GPU batch size for the specified representation
    batch_size = config['batch_size']
    gpu_batch_size = utils.get_gpu_max_batchsize(config['representation'], config['freeze_representation'])
    while batch_size % gpu_batch_size != 0:
        gpu_batch_size-=1 # make sure that the batch_size can be split in equal parts <= max gpu batch size
    grad_accumulation = batch_size // gpu_batch_size

    # (4) Read in datasets
    tokenizer_class = tokenizers[config['representation']]
    tokenizer = tokenizer_class()
    print()
    if config['test_set'] == 'default':
        print('Loading training data...')
        train_set = ir.SingleFastaDataset(dataset_loc=config['training_set'], tokenizer=tokenizer, train_valid_test='train')
        print('Loading validation data...')
        valid_set = ir.SingleFastaDataset(dataset_loc=config['training_set'], tokenizer=tokenizer, train_valid_test='valid')
        print('Loading test data...')
        test_set = ir.SingleFastaDataset(dataset_loc=config['training_set'], tokenizer=tokenizer, train_valid_test='test')
    elif config['test_fold'] >= 0:
        test_folds = {config['test_fold']}
        valid_folds = {(config['test_fold']+1)%10}
        train_folds = set(range(10)) - valid_folds # test proteins are excluded via the exclude_proteins set
        print('Loading test data...')
        test_set = ir.MultiFoldDataset(dataset_loc=config['test_set'],
                                        tokenizer=tokenizer,
                                        exclude_proteins=[],
                                        folds=test_folds)
        exclude_proteins = test_set.get_proteins_in_dataset()
        print('Loading validation data...')
        valid_set = ir.MultiFoldDataset(dataset_loc=config['training_set'],
                                        tokenizer=tokenizer,
                                        exclude_proteins=exclude_proteins,
                                        folds=valid_folds)
        print('Loading training data...')
        train_set = ir.MultiFoldDataset(dataset_loc=config['training_set'],
                                        tokenizer=tokenizer,
                                        exclude_proteins=exclude_proteins,
                                        folds=train_folds)
    else:
        raise AttributeError('If the test set is not set to "default", test_fold needs to be set')
    train_loader = DataLoader(train_set, gpu_batch_size, shuffle=True, pin_memory=True, collate_fn=train_set.collate_fn)
    valid_loader = DataLoader(valid_set, gpu_batch_size, shuffle=False, pin_memory=True, collate_fn=valid_set.collate_fn)
    test_loader = DataLoader(test_set, gpu_batch_size, shuffle=False, pin_memory=True, collate_fn=test_set.collate_fn)


    # (5) Model initialization. Create network architecture, construct pytorch lightning module
    lightning_module = LightningModule(config=config,
                                       tokenizer=tokenizer
    )

    # (6) Set up training
    early_stopping = EarlyStopping(monitor='validation_loss',patience=5,mode='min')
    callbacks = [early_stopping]
    model_dir = 'saved_model/' if config['save_model'] else None
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        accumulate_grad_batches=grad_accumulation,
        logger=logger,
        max_epochs=config['max_epochs'],
        callbacks=callbacks,
        default_root_dir=model_dir,
        val_check_interval=0.25,
    )
    trainer.fit(lightning_module, train_loader, valid_loader)

    # (7) Evaluate predictions on the test set
    trainer.test(dataloaders=test_loader)

