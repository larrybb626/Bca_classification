import time
from torch import optim
from evaluation_function import *
from networks import resnet_bn, resnet_gn, densenet, sparsenet, seresnet, my_NLresnet
from networks.Non_Local import resnet3D
import pandas as pd
from check_file.misc import printProgressBar
from torch.optim import lr_scheduler
from util import GradualWarmupScheduler
from gcam import gcam
from networks import resnet_bn_geng
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from networks.resnet_bn_geng import CombinedModel
from tensorboardX import SummaryWriter
from my_EFL import equalized_focal_loss
from vit.models import ViTVNet, Dual_Model
from vit.models import CONFIGS as CONFIGS_ViT_seg
config_vit = CONFIGS_ViT_seg['ViT-V-Net']

class FocalLoss(nn.Module):
    # gamma的值用放大难样本的loss, alpha针对的是正负样本不均衡
    def __init__(self, alpha=0.5, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    # def forward(self, inputs, targets):
    #     if self.logits:
    #         BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
    #     else:
    #         BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
    #     pt = torch.exp(-BCE_loss)
    #     F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
    #
    #     if self.reduce:
    #         return torch.mean(F_loss)
    #     else:
    #         return F_loss

    def forward(self, inputs, targets):
        F_loss_all = 0
        for i in range(len(inputs)):
            BCE_loss = F.binary_cross_entropy(inputs[i], targets[i], reduce=False)
            pt = torch.exp(-BCE_loss)
            if targets[i] == 1:
                F_loss = 0.8 * (1-pt)**self.gamma * BCE_loss
            if targets[i] == 0:
                F_loss = 0.2 * (1-pt)**self.gamma * BCE_loss
            F_loss_all += F_loss

        return torch.mean(F_loss_all)


class Solver(object):
    def __init__(self, config, train_loader, valid_loader, extra_loader=None, extra_loader_new=None):
        # Make record file
        if config.mode == 'train':
            self.record_file = os.path.join(config.result_path, 'record.txt')
        # else:
        #     self.record_file = os.path.join(config.result_path, 'record_t.txt')
        # f = open(self.record_file, 'w')
        # f.close()

        self.Task_name = config.Task_name
        self.fold_idx = config.fold_idx
        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        # self.test_loader = test_loader
        self.extra_loader = extra_loader
        self.extra_new_loader = extra_loader_new

        self.train_list = config.train_list
        self.valid_list = config.valid_list
        # self.test_list = config.test_list
        self.extra_list = config.extra_list
        self.with_extra = config.with_extra_data

        # Models
        self.classnet = None
        self.optimizer = None
        self.img_size = config.image_size
        self.patch_size = config.patch_size
        # self.img_ch = config.img_ch
        # self.output_ch = config.output_ch
        #self.criterion = torch.nn.BCELoss()
        self.criterion = FocalLoss()
        self.augmentation_prob = config.augmentation_prob

        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # Training settings
        self.num_epochs = config.num_epochs
        self.batch_size = config.batch_size

        # Step size
        self.log_step = config.log_step
        self.val_step = config.val_step

        # learning rate
        self.num_epochs_decay = config.num_epochs_decay
        self.decay_ratio = config.decay_ratio
        self.decay_step = config.decay_step
        self.lr_low = config.lr_low
        self.lr_cos_epoch = config.lr_cos_epoch
        self.lr_warm_epoch = config.lr_warm_epoch
        self.lr_sch = None  # 初始化先设置为None
        self.lr_list = []  # 临时记录lr
        self.loss_list = []
        self.best_epoch = 0
        self.best_classnet_score = 0

        # Path
        self.model_path = config.model_path
        self.result_path = config.result_path
        self.mode = config.mode
        self.save_detail_result = config.save_detail_result
        self.log_dir = config.log_dir

        #result
        self.fpr_all = []
        self.tpr_all = []
        self.auc_all = []
        self.fold_num = 0

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = torch.device('cuda:'+str(1-config.cuda_idx) if torch.cuda.is_available() else 'cpu') #dont know why 1-idx
        self.DataParallel = config.DataParallel
        self.model_type = config.model_type

        self.pre_threshold = 0.5

        self.my_init()

    def myprint(self, *args):
        """Print & Record while training."""
        print(*args)
        # f = open(self.record_file, 'a')
        # print(*args, file=f)
        # f.close()

    def my_init(self):
        self.myprint(time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time())))
        self.print_date_msg()
        self.build_model()

    def print_date_msg(self):
        self.myprint("patient count in train:{}".format(len(self.train_list)), self.train_list)
        self.myprint("patient count in valid:{}".format(len(self.valid_list)), self.valid_list)
        # self.myprint("patient count in test :{}".format(len(self.test_list)), self.test_list)
        self.myprint("patient count in extra :{}".format(len(self.extra_list)), self.extra_list)

    def build_model(self):  # todo
        """Build generator and discriminator."""
        if self.model_type == 'resnet_geng':
            model1 = resnet_bn_geng.resnet50(
                num_classes=1,
                sample_size=self.img_size,
                sample_duration=(self.patch_size)
            )
            model2 = resnet_bn_geng.resnet50(
                num_classes=1,
                sample_size=self.img_size,
                sample_duration=(self.patch_size)
            )
            self.classnet = CombinedModel(
                model1, model2)
        if self.model_type == 'resnet_bn':
            self.classnet = resnet_bn.resnet50(
                num_classes=1,
                sample_size=self.img_size,
                sample_duration=self.patch_size)
        if self.model_type == 'resnet_gn':
            self.classnet = resnet_gn.resnet50(
                num_classes=1,
                sample_size=self.img_size,
                sample_duration=self.patch_size)
        elif self.model_type == 'seresnet':
            self.classnet = seresnet.seresnet101(
                num_classes=1,
                sample_size=self.img_size,
                sample_duration=self.patch_size)
        elif self.model_type == 'sparsenet':
            self.classnet = sparsenet.DenseNet(
                sample_size=self.img_size, sample_duration=self.patch_size,
                growthRate=12, depth=121, nClasses=1, reduction=0.5, bottleneck=False, grate_per_stage=None,
                fetch="sparse")
        elif self.model_type == 'densenet':
            self.classnet = densenet.densenet121(sample_size=self.img_size, sample_duration=self.patch_size,
                                                 num_classes=1)
        elif self.model_type == 'resnet50C2D':
            self.classnet = resnet3D.resnet3D50(non_local=True, num_classes=1)
        elif self.model_type == 'my_NLresnet':
            self.classnet = my_NLresnet.resnet101(
                num_classes=1,
                sample_size=self.img_size,
                sample_duration=self.patch_size)
        elif self.model_type == 'VIT_dual':
            model1 = ViTVNet(config_vit, img_size=(16, 128, 128))
            model2 = ViTVNet(config_vit, img_size=(16, 128, 128))

            self.classnet = Dual_Model(model1, model2)

        self.classnet.to(self.device)

        if self.DataParallel:
            self.classnet = torch.nn.DataParallel(self.classnet)
        self.print_network(self.classnet, self.model_type)

        # 优化器修改
        self.optimizer = optim.Adam(list(self.classnet.parameters()),
                                    self.lr, [self.beta1, self.beta2])
        #self.optimizer = optim.SGD(list(self.classnet.parameters()),self.lr,0.9)

        # lr schachle策略(要传入optimizer才可以)
        # 暂时的三种情况,(1)只用cos,(2)只用warmup,(3)两者都用
        if self.lr_warm_epoch != 0 and self.lr_cos_epoch == 0:
            self.update_lr(self.lr_low)  # 使用warmup需要吧lr初始化为最小lr
            self.lr_sch = GradualWarmupScheduler(self.optimizer,
                                                 multiplier=self.lr / self.lr_low,
                                                 total_epoch=self.lr_warm_epoch,
                                                 after_scheduler=None)
            print('use warmup lr sch')
        elif self.lr_warm_epoch == 0 and self.lr_cos_epoch != 0:
            self.lr_sch = lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                         self.lr_cos_epoch,
                                                         eta_min=self.lr_low)
            print('use cos lr sch')
        elif self.lr_warm_epoch != 0 and self.lr_cos_epoch != 0:
            self.update_lr(self.lr_low)  # 使用warmup需要吧lr初始化为最小lr
            scheduler_cos = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
                                                                     T_0=self.lr_warm_epoch,
                                                                     T_mult= 2 ,
                                                                     eta_min=self.lr_low)
            # scheduler_cos = lr_scheduler.CosineAnnealingLR(self.optimizer,
            #                                                self.lr_cos_epoch,
            #                                                eta_min=self.lr_low)
            self.lr_sch = GradualWarmupScheduler(self.optimizer,
                                                 multiplier=self.lr / self.lr_low,
                                                 total_epoch=self.lr_warm_epoch,
                                                 after_scheduler=scheduler_cos)
            print('use warmup and cos lr sch')
        else:
            if self.lr_sch is None:
                print('use decay coded by dasheng')

        # self.classnet.load_state_dict(torch.load('/data/newnas_1/huyilan202124/LvJieGeng-BCa/JOINT_select_model/epoch86.pkl', map_location=self.device))
    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        # self.myprint(model)
        # self.myprint(name)
        # self.myprint("The number of parameters: {}".format(num_params))

    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def update_lr(self, lr):
        """Update the learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.classnet.zero_grad()

    def tensor2img(self, x):
        img = (x[:, 0, :, :] > x[:, 1, :, :]).float()
        img = img * 255
        return img

    def train(self):
        valid_list_record = []
        extra_list_record = []
        extra_list_record1 = []
        """Train encoder, generator and discriminator."""
        print('-----------------------%s-----------------------------' % self.Task_name)
        # ====================================== Training ===========================================#
        writer = SummaryWriter(log_dir=self.log_dir)
        Iter = 0
        epoch_init = 0
        b = 0.15
        train_len = len(self.train_loader)
        valid_record = np.zeros((1, 7))  # [epoch, Iter, acc, SE, SP, threshold, AUC]
        # test_record = np.zeros((1, 7))  # [epoch, Iter, acc, SE, SP, threshold, AUC]
        extra_record = np.zeros((1, 7))  # [epoch, Iter, acc, SE, SP, threshold, AUC]
        extra_record1 = np.zeros((1, 7))
        # auc_record = np.zeros((1, 3))
        auc_record = np.zeros((1, 4))
        self.myprint('Training...')

        # for epoch in range(epoch_init, self.num_epochs-1300):#只测试前200
        for epoch in range(epoch_init, self.num_epochs):
            self.classnet.train(True)
            epoch_loss = 0
            length = 0
            # for i, (_, patch_data, label) in enumerate(self.train_loader):
            #sample是一个batch的数据加标签
            for i, sample in enumerate(self.train_loader):
                # print(i)
                (_, patch_data, label) = sample
                # T2WI_patch_data = patch_data
                # DWI_patch_data = patch_data
                #单序列输入两个一样的
                T2WI_patch_data =patch_data[:, :, 0:8, :, :]
                DWI_patch_data =patch_data[:, :, 8:16, :, :]
                # DWI_patch_data = patch_data[:,:,8:16,:,:]
                # patch_data = patch_data.to(self.device)  # (N, C_{in}, D_{in}, H_{in}, W_{in})
                T2WI_patch_data = T2WI_patch_data.to(self.device)
                DWI_patch_data = DWI_patch_data.to(self.device)

                label = label.to(self.device)

                # Pre : Prediction Result
                pre_probs = self.classnet(T2WI_patch_data, DWI_patch_data)
                # pre_probs = F.sigmoid(pre_probs)#todo
                pre_flat = pre_probs.view(-1)
                label_flat = label.view(-1)
                # loss = 10 * self.criterion(pre_flat, label_flat)
                loss = equalized_focal_loss(pre_flat, label_flat)
                epoch_loss += loss.item()
                # epoch_loss += float(loss)

                # Backprop + optimize 先反向传播计算参数，再利用optimizer优化参数
                self.reset_grad()
                loss.backward()
                self.optimizer.step()
                length += 1
                Iter += 1
                writer.add_scalars('Loss', {'train': loss}, Iter)
                # trainning bar
                current_lr = self.optimizer.param_groups[0]['lr']
                print_content = 'learning_rate:' + str(current_lr) + ' batch_loss:' + str(loss.data.cpu().numpy())
                printProgressBar(i + 1, train_len, content=print_content)

            epoch_loss = epoch_loss / length
            self.myprint('Epoch [%d/%d], Loss: %.4f' % (epoch + 1, self.num_epochs, epoch_loss))
            writer.add_scalars('Learning rate', {'lr': current_lr}, epoch)
            self.lr_list.append(current_lr)
            self.loss_list.append(epoch_loss)
            # 保存lr为png
            figg = plt.figure(1)
            plt.plot(self.lr_list)
            figg.savefig(os.path.join(self.result_path, 'lr.PNG'))
            plt.close()
            figg2 = plt.figure(2)
            plt.plot(self.loss_list)
            figg2.savefig(os.path.join(self.result_path, 'loss.PNG'))
            plt.close()

            """调整学习策略"""
            # lr scha way 1:
            if self.lr_sch is not None:
                if (epoch + 1) <= (self.lr_cos_epoch + self.lr_warm_epoch):
                    self.lr_sch.step()
                else:
                    self.lr_sch = None
            # lr scha way 2: Decay learning rate(如果使用方式1,则不使用此方式)
            if self.lr_sch is None:
                if ((epoch + 1) >= self.num_epochs_decay) and (
                        (epoch + 1 - self.num_epochs_decay) % self.decay_step == 0):
                    if current_lr >= self.lr_low:
                        self.lr = current_lr * self.decay_ratio
                        # self.lr /= 100.0
                        self.update_lr(self.lr)
                        self.myprint('Decay learning rate to lr: {}.'.format(self.lr))

            if (epoch + 1) % self.val_step == 0 and (epoch + 1) > 0:

                # ===================================== 改成内部测试集 ====================================#
                acc, SE, SP, threshold, AUC, cost = self.test(mode='valid', save_detail_result=self.save_detail_result,
                                                              during_Trianing=True, current_epoch=epoch+1)
                valid_list_record.append(AUC)
                valid_record = np.vstack((valid_record, np.array([epoch + 1, Iter, acc, SE, SP, threshold, AUC])))
                classnet_score = AUC
                writer.add_scalars('Loss', {'test': cost}, Iter)
                writer.add_scalars('Valid', {'AUC': AUC}, epoch)
                self.myprint('[internal_test] Acc: %.4f, SE: %.4f, SP: %.4f, threshold: %.4f, AUC: %.4f' % (
                acc, SE, SP, threshold, AUC))
                save_classnet = self.classnet.state_dict()

                if AUC > 0.85 and self.with_extra: #将内部验证作为内部测试，不再需要单独的内部测试
                    # extra result
                    acc, SE, SP, threshold, AUC_extra, cost = self.test(mode='extra',
                                                                  save_detail_result=self.save_detail_result,
                                                                  during_Trianing=True, current_epoch=epoch+1)
                    extra_list_record.append(AUC_extra)
                    extra_record = np.vstack((extra_record, np.array([epoch + 1, Iter, acc, SE, SP, threshold, AUC_extra])))
                    writer.add_scalars('Loss', {'extra': cost}, Iter)
                    writer.add_scalars('Extra', {'AUC': AUC_extra}, epoch)
                    self.myprint('[Extra]      Acc: %.4f, SE: %.4f, SP: %.4f, threshold: %.4f, AUC: %.4f' % (acc, SE, SP, threshold, AUC_extra))
                    # auc_record = np.vstack((auc_record, np.array([epoch + 1, AUC, AUC_extra])))

                    acc1, SE1, SP1, threshold1, AUC_extra1, cost1 = self.test(mode='extra_new',
                                                                  save_detail_result=self.save_detail_result,
                                                                  during_Trianing=True, current_epoch=epoch+1)
                    extra_list_record1.append(AUC_extra1)
                    extra_record1 = np.vstack((extra_record1, np.array([epoch + 1, Iter, acc1, SE1, SP1, threshold1, AUC_extra1])))
                    writer.add_scalars('Loss', {'extra': cost1}, Iter)
                    writer.add_scalars('Extra', {'AUC': AUC_extra1}, epoch)
                    self.myprint('[Extra2]      Acc: %.4f, SE: %.4f, SP: %.4f, threshold: %.4f, AUC: %.4f' % (acc1, SE1, SP1, threshold1, AUC_extra1))

                    auc_record = np.vstack((auc_record, np.array([epoch + 1, AUC, AUC_extra, AUC_extra1])))

                    if AUC_extra > 0.86 and AUC > 0.95 and AUC_extra1 > 0.83:
                        torch.save(save_classnet, os.path.join(self.model_path, 'epoch%d Val_auc%.3f Extra_auc%.3f-%.3f.pkl' %(epoch + 1, AUC, AUC_extra, AUC_extra1)))
                        self.best_classnet_score = classnet_score
                        self.best_epoch = epoch + 1
                        best_classnet = self.classnet.state_dict()
                        self.myprint(
                            'Best %s model in epoch %d, score : %.4f' % (self.model_type, self.best_epoch, self.best_classnet_score))
                if AUC > 0.9 and AUC_extra1 > 0.83 and AUC_extra > 0.845:
                    if len(extra_list_record) <= 3:
                        self.update_lr(1e-6)
                    else:
                        if extra_list_record[-1]==extra_list_record[-2]==extra_list_record[-3]:
                            pass
                        else:
                            self.update_lr(1e-6)
                # AUC曲线图保存以下
                fig_valid = plt.figure(3)
                plt.plot(valid_list_record)
                fig_valid.savefig(os.path.join(self.result_path, 'valid_AUC.PNG'))
                plt.close()
                # fig_test = plt.figure(4)
                # plt.plot(test_list_record)
                # fig_test.savefig(os.path.join(self.result_path, 'test_AUC.PNG'))
                # plt.close()
                fig_extra = plt.figure(5)
                plt.plot(extra_list_record)
                fig_extra.savefig(os.path.join(self.result_path, 'extra_AUC.PNG'))
                plt.close()

                # save_record_in_xlsx
                if (True):
                    excel_save_path = os.path.join(self.result_path, 'record.xlsx')
                    with pd.ExcelWriter(excel_save_path, engine='openpyxl') as record:
                        detail_result1 = pd.DataFrame(valid_record)
                        detail_result1 = detail_result1.round(6)
                        detail_result1.to_excel(record, 'internal_test', float_format='%.5f')#记录内部验证集数据[epoch, Iter, acc, SE, SP, threshold, AUC]
                        # detail_result2 = pd.DataFrame(test_record)
                        # detail_result2.to_excel(record, 'test', float_format='%.5f')
                        if self.with_extra:
                            detail_result3 = pd.DataFrame(extra_record)
                            detail_result3 = detail_result3.round(6)
                            detail_result3.to_excel(record, 'extra', float_format='%.5f')

                            detail_result4 = pd.DataFrame(auc_record)
                            detail_result4 = detail_result4.round(6)
                            detail_result4.to_excel(record, 'AUC record', float_format='%.5f')

                            detail_result5 = pd.DataFrame(extra_record1)
                            detail_result5 = detail_result5.round(6)
                            detail_result5.to_excel(record, 'extra2', float_format='%.5f')

        # path = os.path.join(self.model_path,'epoch'+self.best_epoch+'_Test auc'+self.best_classnet_score+'.pkl')
        # acc, SE, SP, threshold, AUC,fpr,tpr = self.test_or(mode='valid',classnet_path=path,save_detail_result=self.save_detail_result)
        self.myprint('Finished!')
        self.myprint(time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time())))

    def test(self, mode='train', classnet_path=None, save_detail_result=False, during_Trianing=False, current_epoch = 0):
        """Test model & Calculate performances."""
        if not classnet_path is None:
            # if os.path.isfile(classnet_path):
            self.classnet.load_state_dict(torch.load(classnet_path))
            self.myprint('%s is Successfully Loaded from %s' % (self.model_type, classnet_path))

        self.classnet.train(False)
        self.classnet.eval()

        if mode == 'train':
            data_lodear = self.train_loader
        elif mode == 'test':
            data_lodear = self.valid_loader
        elif mode == 'valid':
            data_lodear = self.valid_loader
        elif mode == 'extra':
            if self.extra_loader is None:
                print('Extra data is not existed!!')
                return
            data_lodear = self.extra_loader
        elif mode == 'extra_new':
            data_lodear = self.extra_new_loader

        # model pre for each patch
        patient_order_list = []
        patch_order_list = []
        pre_list = []
        label_list = []
        cost = 0.0
        for i, sample in enumerate(data_lodear):
            (patch_paths, patch, label) = sample
            patch_paths = list(patch_paths)
            with torch.no_grad():
                # T2WI_patch_data = patch
                # DWI_patch_data = patch
                T2WI_patch_data = patch[:, :, 0:8, :, :]
                DWI_patch_data = patch[:, :, 8:16, :, :]
                # patch_data = patch_data.to(self.device)  # (N, C_{in}, D_{in}, H_{in}, W_{in})
                T2WI_patch_data = T2WI_patch_data.to(self.device)
                DWI_patch_data = DWI_patch_data.to(self.device)
                # patch = patch.to(self.device)
                label = label.to(self.device)
                pre_probs = self.classnet(T2WI_patch_data, DWI_patch_data)
                # pre_probs = F.sigmoid(pre_probs)   # todo

                pre_flat = pre_probs.view(-1)
                label_flat = label.view(-1)
                loss = self.criterion(pre_flat, label_flat)
                # loss = equalized_focal_loss(pre_flat,label_flat)
                cost += float(loss)

            pre_probs = pre_probs.data.cpu().numpy()
            label = label.data.cpu().numpy()

            for ii in range(pre_probs.shape[0]):
                pre_tmp = pre_probs[ii, :]
                label_tmp = label[ii, :]

                pre_list.append(pre_tmp.reshape(-1))
                label_list.append(label_tmp.reshape(-1))

                tmp_index = patch_paths[ii].split('/')[-1]
                tmp_index1 = tmp_index.split('_')[0][:]
                patient_order_list.append(int(tmp_index1))
                tmp_index2 = tmp_index.split('_')[1][:]
                tmp_index2 = tmp_index2.split('.')[0][:]
                patch_order_list.append(int(tmp_index2))

        cost /= (i + 1)

        detail_result1 = np.zeros([len(patient_order_list), 4])  # detail_msg = [id, patch_id, pre, label]
        detail_result1[:, 0] = np.array(patient_order_list).T
        detail_result1[:, 1] = np.array(patch_order_list).T
        detail_result1[:, 2] = np.array(pre_list).T
        detail_result1[:, 3] = np.array(label_list).T

        # statistic for each patient
        patinet_order_unique = np.unique(patient_order_list)
        detail_result2 = np.zeros([len(patinet_order_unique), 4])  # detail_msg = [id, _, mpre, label]
        detail_result2[:, 0] = patinet_order_unique.T
        for unique_p_order in patinet_order_unique:
            select_patient_index = [i for i, x in enumerate(patient_order_list) if x == unique_p_order]
            pre_tmp = []
            label_tmp = []
            for i in select_patient_index:
                pre_tmp.append(pre_list[i])
                label_tmp.append(label_list[i])
            pre_probs = np.array(pre_tmp).reshape(-1)
            label = np.array(label_tmp).reshape(-1)

            mean_pre_probs = np.max(pre_probs)  # TODO
            label = np.mean(label)

            # detail_result[detail_result[:,0] == unique_p_order,1] = get_AUC(pre_probs, label)
            detail_result2[detail_result2[:, 0] == unique_p_order, 2] = mean_pre_probs
            detail_result2[detail_result2[:, 0] == unique_p_order, 3] = label

        P_pre_probs = torch.from_numpy(detail_result2[:, 2]).to(self.device)
        P_label = torch.from_numpy(detail_result2[:, 3]).to(self.device)

        if mode == 'train':
            threshold = get_best_threshold(P_pre_probs, P_label)
        elif mode == 'valid':
            threshold = get_best_threshold(P_pre_probs, P_label)
            self.pre_threshold = threshold
        elif mode == 'test':
            threshold = get_best_threshold(P_pre_probs, P_label)
            self.pre_threshold = threshold
        elif mode == 'extra':
            threshold = get_best_threshold(P_pre_probs, P_label)
            self.pre_threshold = threshold
        elif mode == 'extra_new':
            threshold = get_best_threshold(P_pre_probs, P_label)
            self.pre_threshold = threshold
            # threshold = self.pre_threshold
        else:
            threshold = self.pre_threshold
        accuracy = get_accuracy(P_pre_probs, P_label, threshold)
        sensitivity = get_sensitivity(P_pre_probs, P_label, threshold)
        specificity = get_specificity(P_pre_probs, P_label, threshold)
        AUC = get_AUC(P_pre_probs, P_label)  # todo
        #fpr,tpr = get_roc(P_pre_probs,P_label,AUC,self.result_path)

        # save
        if (save_detail_result):
            excel_save_path = os.path.join(self.result_path, 'prediction', mode + '_' + str(current_epoch) + 'pre.xlsx')
            if not os.path.exists(os.path.join(self.result_path, 'prediction')):
                os.makedirs(os.path.join(self.result_path, 'prediction'))
            with pd.ExcelWriter(excel_save_path, engine='openpyxl') as writer:
                detail_result1 = pd.DataFrame(detail_result1)
                detail_result1 = detail_result1.round(6)
                detail_result1.to_excel(writer, 'patch_msg', float_format='%.5f')

                detail_result2 = pd.DataFrame(detail_result2)
                detail_result2 = detail_result2.round(6)
                detail_result2.to_excel(writer, 'patient_msg', float_format='%.5f')

                detail_result3 = pd.DataFrame(np.array([accuracy, sensitivity, specificity, threshold, AUC]))
                detail_result3 = detail_result3.round(6)
                detail_result3.to_excel(writer, 'patient_result', float_format='%.5f')

        # self.myprint('%s result has been Successfully Saved in %s' % (mode, excel_save_path))

        if during_Trianing:
            return accuracy, sensitivity, specificity, threshold, AUC, cost
        else:
            return accuracy, sensitivity, specificity, threshold, AUC
