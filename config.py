import os

config = {
    # General
    'cwd_path': os.getcwd(),
    # 'mxnet_path': '~/.conda/envs/lcnn-3.6.0/lib/python3.6/site-packages/mxnet',
   'mxnet_path': '~/incubator-mxnet', 

    # Data records
    'train_db': 'data/train.rec',
    'val_db': 'data/test.rec',
    'num_classes': 2,

    # Data preprocessing and augmentation
    'data_params': {
        'mean': [0.0, 0.0, 0.0],
        'shuffle': True,
        'rand_crop': False,
        'rand_mirror': True,
    },
    
    # Snapshotting params
    'experiment_dir': 'experiments/experiment_dir',
    'model_prefix': 'net',

    # Whether use pretrained model as init
    'init_from_trained': False,
    'init_model': ['path-to-pretrained-network/net', 100],

    # Input params
    'input_shape': (3, 128, 128),
    'batch_size': 128,

    # Optimizer param
    'lr_base': 0.00025,
    'lr_step': 500,
    'lr_factor': 0.1,
    'momentum': 0.9,
    'wd': 0.01,

    # Iteration params
    'epoch_size': 100,
    'begin_epoch': 0,
    'num_epoch': 10,
    'display': 100,

    # GPU devices
    'devices_id': [0]
}
