import warnings
warnings.filterwarnings('ignore')
import argparse
from solver import Solver
from data_loader_function import get_loader
from torch.backends import cudnn
import random
from evaluation_function import *
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import h5py
import os

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


aucs = []
fprs = []
tprs = []
def main(config, classnet_path=None):

    """create the folder we needed"""
    cudnn.benchmark = True
    timestr = time.strftime("%Y%m%d-%H%M%S")
    # config.Task_name = config.Task_name + '_fold' + str(config.fold_idx)
    config.Task_name = config.Task_name + '_fold' + timestr
    config.result_path = os.path.join(config.result_path, config.Task_name)
    config.model_path = os.path.join(config.result_path, 'models')
    config.log_dir = os.path.join(config.result_path, 'logs')
    # Create directories if not exist
    if config.mode == 'train':
        if not os.path.exists(config.result_path):
            os.makedirs(config.result_path)
            os.makedirs(config.model_path)
            os.makedirs(config.log_dir)

        print(config)
        f = open(os.path.join(config.result_path, 'config.txt'), 'w')
        for key in config.__dict__:
            print('%s: %s' % (key, config.__getattribute__(key)), file=f)
        f.close()

    """The setting of the test mode"""
    if config.mode == 'test_cam':
        config.batch_size = 1
        config.batch_size_test = 1
        config.augmentation_prob = 0.

    """The setting of the cuda used"""
    if not config.DataParallel:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config.cuda_idx)
    # else:
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


    # # make_cross_fold
    """Make training and validation dataset"""
    data_all_list = os.listdir(config.h5data_path)
    data_all_list.sort()
    random.seed(2022)   # 2020
    random.shuffle(data_all_list)
    patient_num = len(data_all_list)

    x_list = []
    y_list = []
    for counter in data_all_list:
        data_path = os.path.join(config.h5data_path, counter, counter.split("_")[0] + "_1.h5")
        h5_data = h5py.File(data_path)
        h5_label = h5_data["Label"]
        x_list.append(counter)
        y_list.append(h5_label)
    print("y_sum:{}".format(np.sum(y_list)))

    train_list_all = []
    val_list_all = []
    # kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=fold_i)
    # seed = config.fold_idx+2026#每次都更新随机种子,上一次2026
    # ss_geng = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=2026)#之前的889是2026的随机种子，3047--0.880
    ss_geng = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=config.fold_seed)#之前的889是2026的随机种子，3047--0.880
    # print("fold_i",fold_i)
    for train_index, val_index in ss_geng.split(x_list, y_list):
        num_train_pos = np.sum([y_list[index] for index in train_index])
        # print('train_index_sum:', np.sum(num_train_pos))
        num_test_pos = np.sum([y_list[index] for index in val_index])
        # print('test_index_sum', np.sum(num_test_pos))
        train_index_str = [str(x_list[index]) for index in train_index]
        # print("train_index_str:{}".format(train_index_str))
        val_index_str = [str(x_list[index]) for index in val_index]
        # print("val_index_str:{}".format(val_index_str))

        train_list_all.append(train_index_str)
        val_list_all.append(val_index_str)

    train_list = list(train_list_all[1])
    valid_list = list(val_list_all[1])

    config.train_list = train_list
    print("train_list:{}".format(train_list))
    config.valid_list = valid_list
    print("valid_list:{}".format(valid_list))

    """Read the test data"""
    # test_list = os.listdir(config.inter_h5data_path)
    # test_list.sort(key=lambda x: int(x[:]))
    # config.test_list = test_list
    # print("test_list:{}".format(test_list))

    """Packing the data"""
    train_loader = get_loader(image_root=config.h5data_path,
                              image_list=train_list,
                              image_size=config.image_size,
                              batch_size=config.batch_size,
                              num_workers=config.num_workers,
                              mode='train',
                              augmentation_prob=config.augmentation_prob)
    valid_loader = get_loader(image_root=config.h5data_path,
                              image_list=valid_list,
                              image_size=config.image_size,
                              batch_size=config.batch_size_test,
                              num_workers=config.num_workers,
                              mode='valid',
                              augmentation_prob=0.)
    # test_loader = get_loader(image_root=config.inter_h5data_path,
    #                          image_list=test_list,
    #                          image_size=config.image_size,
    #                          batch_size=config.batch_size_test,
    #                          num_workers=config.num_workers,
    #                          mode='test',
    #                          augmentation_prob=0.)

    """Read the extra data"""
    if config.with_extra_data:
        extra_list = os.listdir(config.extra_h5data_path)
        extra_list.sort(key=lambda x: int(x[:]))
        print("extra_list:{}".format(extra_list))
        config.extra_list = extra_list
        extra_loader = get_loader(image_root=config.extra_h5data_path,
                                  image_list=extra_list,
                                  image_size=config.image_size,
                                  batch_size=config.batch_size_test,
                                  num_workers=config.num_workers,
                                  mode='extra',
                                  augmentation_prob=0.)
    else:
        config.extra_list = []
        extra_loader = None

    """The package of the data"""
    # solver = Solver(config, train_loader, valid_loader, test_loader, extra_loader)
    solver = Solver(config, train_loader, valid_loader, extra_loader)
    # Train and sample the images
    if config.mode == 'train':
        solver.train()
    elif config.mode == 'test':
        # classnet_path = os.path.join(config.model_path, 'epoch1464 Val_auc0.908 Test_auc0.963 Extra_auc0.861.pkl')
        classnet_path = "/nfs-data-new/LJG/BCa_classification/pick-op-model/epoch752 Val_auc0.918 Extra_auc0.849.pkl"
        # classnet_path = os.path.join(config.model_path, 'best_classnet_score.pkl')

        """Another testing method ———— test_or"""
        # test_or
        # mytest = solver.test_or
        # acc, SE, SP, threshold, AUC = mytest(mode='train', classnet_path = classnet_path, save_detail_result = False)
        # print('[Training  ] Acc: %.4f, SE: %.4f, SP: %.4f, threshold: %.4f, AUC: %.4f' % (acc, SE, SP, threshold, AUC))
        # acc, SE, SP, threshold, AUC = mytest(mode='valid', classnet_path = classnet_path, save_detail_result = False)
        # print('[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, threshold: %.4f, AUC: %.4f' % (acc, SE, SP, threshold, AUC))
        # acc, SE, SP, threshold, AUC = mytest(mode='test', classnet_path = classnet_path, save_detail_result = False)
        # print('[Testing]    Acc: %.4f, SE: %.4f, SP: %.4f, threshold: %.4f, AUC: %.4f' % (acc, SE, SP, threshold, AUC))
        # acc, SE, SP, threshold, AUC = mytest(mode='extra', classnet_path = classnet_path, save_detail_result = False)
        # print('[Extra]      Acc: %.4f, SE: %.4f, SP: %.4f, threshold: %.4f, AUC: %.4f' % (acc, SE, SP, threshold, AUC))

        """The testing method I used"""
        mytest = solver.test
        acc, SE, SP, threshold, AUC = mytest(mode='extra', classnet_path=classnet_path, save_detail_result=True)
        print('[Testing]    Acc: %.4f, SE: %.4f, SP: %.4f, threshold: %.4f, AUC: %.4f' % (acc, SE, SP, threshold, AUC))

    elif config.mode == 'test_or':
        """Loading the model"""
        model_name = str(config.fold_idx) + ".pkl"
        classnet_path = os.path.join('/media/root/3339482d-9d23-44ee-99a0-85e517217d153/CKY/2022_all_new_data_36/result/best_models_QZX', model_name)
        print(classnet_path)

        """Begining to test"""
        mytest = solver.test_or
        time_start = time.time()
        acc, SE, SP, threshold, AUC, fpr, tpr = mytest(mode='extra', classnet_path=classnet_path, save_detail_result=True)
        time_end = time.time()

        """Satisfied the results"""
        aucs.append(AUC)
        tprs.append(tpr)
        fprs.append(fpr)
        print('[Testing]    Acc: %.4f, SE: %.4f, SP: %.4f, threshold: %.4f, AUC: %.4f' % (acc, SE, SP, threshold, AUC))
        print('totally cost', time_end-time_start)

    elif config.mode == 'test_cam':
        """The method to visual the heatmap"""
        classnet_path = os.path.join(config.model_path, 'epoch980_Testdice0.6429.pkl')
        mytest = solver.test_cam
        mytest(mode='valid', classnet_path=classnet_path, save_detail_result=False)


def seed_torch(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


if __name__ == '__main__':
    # for r in range(15):
    # for fold_i in[2026,3047,4098,4399,2020]:
    # for fold_i in[2026]:
        # for i in [2023, 2012, 2022, 2032, 3407]:
            # flod_i = 2026
            seed_torch(seed=4011)  # set random seed for whole environment  2013
            parser = argparse.ArgumentParser()
            parser.add_argument('--fold_seed', type=int, default=2026, help='folds random seed')
            parser.add_argument('--fold_K', type=int, default=5, help='folds number after divided')
            parser.add_argument('--fold_idx', type=int, default=2)

            # model hyper-parameters
            parser.add_argument('--image_size', type=int, default=128)
            parser.add_argument('--patch_size', type=int, default=8)

            # training hyper-parameters
            parser.add_argument('--img_ch', type=int, default=1)
            parser.add_argument('--output_ch', type=int, default=1)
            parser.add_argument('--num_epochs', type=int, default=1500)
            parser.add_argument('--batch_size', type=int, default=24)
            parser.add_argument('--batch_size_test', type=int, default=32)
            parser.add_argument('--num_workers', type=int, default=32)  # !!!dont change!!!
            parser.add_argument('--lr', type=float, default=0.0001)
            parser.add_argument('--beta1', type=float, default=0.0045)  # momentum1 in AdamY

            parser.add_argument('--beta2', type=float, default=0.9999)  # momentum2 in Adam
            parser.add_argument('--augmentation_prob', type=float, default=1.0) 

            parser.add_argument('--log_step', type=int, default=1)
            parser.add_argument('--val_step', type=int, default=1)
            parser.add_argument('--num_epochs_decay', type=int, default=200)  # decay开始的最小epoch数
            parser.add_argument('--decay_ratio', type=float, default=0.95)  # 0~1,每次decay到1*ratio
            parser.add_argument('--decay_step', type=int, default=40)  # epoch
            parser.add_argument('--lr_low', type=float, default=1e-6)  # 最小学习率,设置为None,则为最大学习率的1e+6分之一(不可设置为0)
            parser.add_argument('--lr_warm_epoch', type=int, default=45)  # warmup的epoch数,一般就是10~20,为0或False则不使用
            parser.add_argument('--lr_cos_epoch', type=int, default=1480)  # cos退火的epoch数,一般就是总epoch数-warmup的数,为0或False则代表不使用

            """training mode selected"""
            parser.add_argument('--mode', type=str, default='train', help='train/test/test_cam')
            parser.add_argument('--model_type', type=str, default='resnet_geng',  # resnet_bn
                                help='resnet/densenet/sparsenet/seresnet/resnet50C2D/my_NLresnet')
            parser.add_argument('--Task_name', type=str, default='final_ResNet_classify', help='DIR name,Task name')
            parser.add_argument('--cuda_idx', type=int, default=1)  # 设置跑哪条卡
            parser.add_argument('--DataParallel', type=bool, default=False)

            # # """双序列数据集,我制作的内部训练集没毛病"""
            parser.add_argument('--h5data_path', type=str,
                                default='/data/newnas/huyilan2021124/BCa_DATA/dual_train_h5')  # arterial venous /data/newnas/huyilan2021124/BCa_DATA/dual_train_h5
            parser.add_argument('--with_extra_data', type=bool, default=True)
            parser.add_argument('--extra_h5data_path', type=str,
                                default='/data/newnas/huyilan2021124/BCa_DATA/dual_ex_h51')  # arterial venous /data/newnas/huyilan2021124/BCa_DATA/dual_ex_h51
             # """T2WI数据集,我制作的内部训练集没毛病"""
            # parser.add_argument('--h5data_path', type=str,
            #                     default='/home/user14/sharedata/newnas/LJG/BCa_DATA/data/T2WI_h5_data_all_train')  # arterial venous
            # parser.add_argument('--with_extra_data', type=bool, default=True)
            # parser.add_argument('--extra_h5data_path', type=str,
            #                     default='/home/user14/sharedata/newnas/LJG/BCa_DATA/new-t2wi-ex-h5/h5_data20230428-225211')  # arterial venous
            """results path setting"""
            # parser.add_argument('--result_path', type=str, default='/nfs-data-new/LJG/BCa_classification/Result')
            parser.add_argument('--result_path', type=str, default='/data/newnas_1/huyilan202124/LvJieGeng-BCa/Joint_Result/2-27')  #  /data/newnas_1/huyilan202124/LvJieGeng-BCa/Joint_Result
            parser.add_argument('--save_detail_result', type=bool, default=True)
            config = parser.parse_args()

            """Begining to run"""
            main(config)
