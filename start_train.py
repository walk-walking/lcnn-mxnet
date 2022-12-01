import logging, sys, os
import json

from config import config
from utils.net_train import train_net
from utils.data_utils import *

def save_config(cfg):
    save_fname = cfg['experiment_dir'] + '/config.json'  #experiments/casia/scale_2.2__fold_1/config.json
    with open(save_fname, 'w') as f:
        json.dump(cfg, f, sort_keys=True, indent=4)

def set_logging(experiment_dir):
    # duplicate logging to file and stdout
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s]\t%(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=experiment_dir + '/log.txt',  #experiments/casia/scale_2.2__fold_1/log.txt
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logging.getLogger().addHandler(console)

def start_experiment(cfg):
    if not os.path.exists(cfg['experiment_dir']):
        os.makedirs(cfg['experiment_dir'])
    save_config(cfg)
    set_logging(cfg['experiment_dir'])
    train_net(cfg)

# k为[1,5]之间的整数 
def train_net_casia(data_dir, scale, k):
    # client_num = 5  
    client_num = 20  #单个尺寸下数据集的文件夹个数
    folds_num = 5  #用于dev的文件夹个数
    client_per_fold = int(client_num / folds_num) # 转成int 否则45行报错

    list_dir = data_dir + '/mxnet'  # data/casia/mxnet
    db_dir = data_dir + '/scale_' + str(scale) + '/train_release'   # data/casia/scale_2.2/train_release
    experiment_str = 'scale_' + str(scale) + '__fold_' + str(k)  #scale_2.2__fold_1

    # prepare dev data
    client_list_dev = [i for i in range((k - 1) * client_per_fold + 1, k * client_per_fold + 1)]  #[1,2,3,4]  or [5,6,7,8] 对应的是train_release目录下1、2、3、4  当前目录下只有5 所以k=3时就没有数据了
    # client_list_dev =  [k,k+2]  # k取1、2、3  [1,3] [2,4]、[3,5] 

    dev_list_fname = list_dir + '/' + experiment_str + '__dev'  # data/casia/mxnet/scale_2.2__fold_1__dev
    #生成data/casia/mxnet/scale_2.2__fold_1__dev.lst 
    #内容格式为 0       1.000000        data/casia/scale_2.2/train_release/1/1/frame_0.jpg
    create_imglist_casia(db_dir, dev_list_fname, False, client_list_dev)  
    #将图片打包为rec格式 生成data/casia/mxnet/scale_2.2__fold_1__dev.rec 
    create_record_file(config['mxnet_path'], config['cwd_path'], dev_list_fname)

    # prepare train data
    client_list_train = list(set(range(1, client_num + 1)) - set(client_list_dev)) #train_release目录下非dev的全部用在train 
    train_list_fname = list_dir + '/' + experiment_str + '__train'
    # 生成data/casia/mxnet/scale_2.2__fold_1__train.lst
    #内容格式为  170     0.000000        data/casia/scale_2.2/train_release/5/3/frame_20.jpg
    create_imglist_casia(db_dir, train_list_fname, True, client_list_train)
    # data/casia/mxnet/scale_2.2__fold_1__train.rec
    create_record_file(config['mxnet_path'], config['cwd_path'], train_list_fname)

    # set config
    config['train_db'] = train_list_fname + '.rec'  # data/casia/mxnet/scale_2.2__fold_1__train.rec
    config['val_db'] = dev_list_fname + '.rec' # data/casia/mxnet/scale_2.2__fold_1__dev.rec

    with open('%s/mean_std__scale_%s.txt' % (data_dir, str(scale)), 'r') as f: # data/casia/mean_std_scale_2.2.txt   计算当前使用数据集的结果所得
        config['data_params']['mean'] = [float(item) for item in f.readline().split()] 
    
    config['experiment_dir'] = 'experiments/casia/' + experiment_str  #experiments/casia/scale_2.2__fold_1 

    # start experiment
    start_experiment(config)


def main(argc, argv):
    # print(argv[1]) # scale  2.2
    # print(argv[2]) # fold  1/2/3/4
    train_net_casia('data/casia', float(argv[1]), int(argv[2]))


if __name__ == "__main__":
    main(len(sys.argv), sys.argv)
