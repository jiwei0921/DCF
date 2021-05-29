import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pdb, os, argparse
from tqdm import tqdm
from datetime import datetime
from model.DCF_models import DCF_VGG
from model.DCF_ResNet_models import DCF_ResNet
from model.depth_calibration_models import discriminator, depth_estimator
from utils import clip_gradient, adjust_lr, iou
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from evaluateSOD.main import evalateSOD
from data import get_loader,test_dataset,calibration_dataset,get_calibration_loader
from tqdm import trange
from skimage import io
from torch.autograd import Variable
import shutil


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
cudnn.benchmark = True


writer = SummaryWriter()
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt_load', type=bool, default=False, help='whether load checkpoint or not')
parser.add_argument('--discriminator_load', type=bool, default=False, help='whether load checkpoint in discriminator ')
parser.add_argument('--snapshot', type=int, default=110, help='load checkpoint number')
parser.add_argument('--calib_flag', type=int, default=135, help='depth calibration begin epoch')
parser.add_argument('--is_ResNet', type=bool, default=True, help='VGG or ResNet backbone')

parser.add_argument('--epoch', type=int, default=200, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=60, help='every n epochs decay learning rate')
opt = parser.parse_args()
print('Learning Rate: {} ResNet: {}'.format(opt.lr, opt.is_ResNet))


image_root = '/Data/train_ori/train_images/'
gt_root = '/Data/train_ori/train_masks/'
depth_root = '/Data/train_ori/train_depth/'
val_root = '/Data/test_data/'
validation = ['LFSD']



train_loader = get_loader(image_root, gt_root, depth_root, batchsize=opt.batchsize, trainsize=opt.trainsize)

# build models
if opt.is_ResNet:
    model_rgb = DCF_ResNet()
    model_depth = DCF_ResNet()
    model_discriminator = discriminator(n_class=2)
    model_estimator = depth_estimator()
    if opt.ckpt_load:
        model_rgb.load_state_dict(torch.load('./ckpt/DCF_Resnet/' + 'DCF_rgb.pth.' + str(opt.snapshot)))
        model_depth.load_state_dict(torch.load('./ckpt/DCF_Resnet/' + 'DCF_depth.pth.' + str(opt.snapshot)))
    if opt.discriminator_load:
        model_discriminator.load_state_dict(torch.load('./ckpt/DCF_Resnet/' + 'DCF_dis.pth.' + str(opt.snapshot)))
        model_estimator.load_state_dict(torch.load('./ckpt/DCF_Resnet/' + 'DCF_estimator.pth.' + str(opt.snapshot)))
else:
    model_rgb = DCF_VGG()
    model_depth = DCF_VGG()
    model_discriminator = discriminator(n_class=2)
    model_estimator = depth_estimator()
    if opt.ckpt_load:
        model_rgb.load_state_dict(torch.load('./ckpt/DCF_VGG/' + 'DCF_rgb.pth.' + str(opt.snapshot)))
        model_depth.load_state_dict(torch.load('./ckpt/DCF_VGG/' + 'DCF_depth.pth.' + str(opt.snapshot)))
    if opt.discriminator_load:
        model_discriminator.load_state_dict(torch.load('./ckpt/DCF_VGG/' + 'DCF_dis.pth.' + str(opt.snapshot)))
        model_estimator.load_state_dict(torch.load('./ckpt/DCF_Resnet/' + 'DCF_estimator.pth.' + str(opt.snapshot)))



cuda = torch.cuda.is_available()
if cuda:
    model_rgb.cuda()
    model_depth.cuda()
    model_discriminator.cuda()
    model_estimator.cuda()

params_rgb = model_rgb.parameters()
params_depth = model_depth.parameters()
params_dis = model_discriminator.parameters()
params_estimator = model_estimator.parameters()

optimizer_rgb = torch.optim.Adam(params_rgb, opt.lr)
optimizer_depth = torch.optim.Adam(params_depth, opt.lr)
optimizer_dis = torch.optim.Adam(params_dis, opt.lr)
optimizer_estimator = torch.optim.Adam(params_estimator,opt.lr)

total_step = len(train_loader)
CE = torch.nn.BCEWithLogitsLoss()
TML = torch.nn.TripletMarginLoss(margin=1.0, p=2.0, eps=1e-6, swap=False, reduction='mean')
smo_L1 = torch.nn.SmoothL1Loss()
cross_E = torch.nn.CrossEntropyLoss()

def train(train_loader, model_rgb, model_depth,
          optimizer_rgb, optimizer_depth, epoch):
    model_rgb.train()
    model_depth.train()

    for i, pack in enumerate(tqdm(train_loader), start=1):
        iteration = i + epoch*len(train_loader)

        optimizer_rgb.zero_grad()
        optimizer_depth.zero_grad()

        images, gts, depths = pack
        images = Variable(images)
        gts = Variable(gts)
        depths = Variable(depths)
        if cuda:
            images = images.cuda()
            gts = gts.cuda()
            depths = depths.cuda()


        '''~~~Your Framework~~~'''
        # RGB Stream
        atts_rgb, dets_rgb,x3_r,x4_r,x5_r = model_rgb(images)
        loss1_rgb = CE(atts_rgb, gts)
        loss2_rgb = CE(dets_rgb, gts)
        loss_rgb = (loss1_rgb + loss2_rgb) / 2.0
        loss_rgb.backward()

        clip_gradient(optimizer_rgb, opt.clip)
        optimizer_rgb.step()

        # Depth Stream
        depths = torch.cat([depths,depths,depths],dim=1)
        atts_depth, dets_depth, x3_d,x4_d,x5_d = model_depth(depths)
        loss1_depth = CE(atts_depth, gts)
        loss2_depth = CE(dets_depth, gts)
        loss_depth = (loss1_depth + loss2_depth) / 2.0
        loss_depth.backward()

        clip_gradient(optimizer_depth, opt.clip)
        optimizer_depth.step()

        '''~~~END~~~'''


        if i % 400 == 0 or i == total_step:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss_rgb: {:.4f} Loss_depth: {:0.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss_rgb.data, loss_depth.data))
        writer.add_scalar('Loss/rgb', loss_rgb.item(), iteration)
        writer.add_scalar('Loss/depth', loss_depth.item(), iteration)
        writer.add_images('Results/rgb', dets_rgb.sigmoid(), iteration)
        writer.add_images('Results/depth', dets_depth.sigmoid(), iteration)

    if opt.is_ResNet:
        save_path = 'ckpt/DCF_Resnet/'
    else:
        save_path = 'ckpt/DCF_VGG/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if (epoch+1) % 5 == 0:
        torch.save(model_rgb.state_dict(), save_path + 'DCF_rgb.pth' + '.%d' % (epoch+1))
        torch.save(model_depth.state_dict(), save_path + 'DCF_depth.pth' + '.%d' % (epoch + 1))




def train_calibration(calibration_loader, model_rgb,model_depth, model_discriminator, model_estimator,
          optimizer_dis, optimizer_estimator, epoch,key_min):
    model_rgb.eval()
    model_depth.eval()
    model_discriminator.train()
    model_estimator.train()

    #for i in trange(calibration_loader.size):
    for i , pack in enumerate(tqdm(calibration_loader),start=1):
        iteration = i + epoch * len(calibration_loader)

        optimizer_dis.zero_grad()
        optimizer_estimator.zero_grad()

        # images, gts, depths, name = calibration_loader.load_data()
        images, gts, depths, name = pack
        images = Variable(images)
        gts = Variable(gts)
        depths = Variable(depths)
        cuda = torch.cuda.is_available()

        if cuda:
            images = images.cuda()
            gts = gts.cuda()
            depths = depths.cuda()

        '''~~~Your Framework~~~'''
        # RGB Stream
        with torch.no_grad():
            if epoch == opt.calib_flag:
                if opt.is_ResNet:
                    model_rgb.load_state_dict(torch.load('./ckpt/DCF_Resnet/' + 'DCF_rgb.pth' + key_min))
                    model_depth.load_state_dict(torch.load('./ckpt/DCF_Resnet/' + 'DCF_depth.pth' + key_min))
                else:
                    model_rgb.load_state_dict(torch.load('./ckpt/DCF_VGG/' + 'DCF_rgb.pth' + key_min))
                    model_depth.load_state_dict(torch.load('./ckpt/DCF_VGG/' + 'DCF_depth.pth' + key_min))
            if cuda:
                model_rgb.cuda()
            _, _,x3_r,x4_r,x5_r = model_rgb(images)

        x3_, x4_, x5_ = x3_r.detach(), x4_r.detach(), x5_r.detach()

        pred_depth = model_estimator(images,x3_,x4_,x5_)
        loss_dep = smo_L1(pred_depth, depths)
        loss_dep.backward()
        clip_gradient(optimizer_estimator, opt.clip)
        optimizer_estimator.step()


        score = model_discriminator(depths)
        loss_dis = cross_E(score,gts.squeeze())
        loss_dis.backward()
        clip_gradient(optimizer_dis, opt.clip)
        optimizer_dis.step()
        '''~~~END~~~'''



        if i % 20 == 0:
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss_dis: {:.4f} Loss_estimator: {:0.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss_dis.data, loss_dep.data))
        writer.add_scalar('Loss/discriminator', loss_dis.item(), iteration)
        writer.add_scalar('Loss/estimator', loss_dep.item(), iteration)
        writer.add_images('Results/pred_depth', pred_depth, iteration)

    if opt.is_ResNet:
        save_path = 'ckpt/DCF_Resnet/'
    else:
        save_path = 'ckpt/DCF_VGG/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if (epoch+1) % 5 == 0:
        torch.save(model_discriminator.state_dict(), save_path + 'DCF_dis.pth' + '.%d' % (epoch+1))
        torch.save(model_estimator.state_dict(), save_path + 'DCF_estimator.pth' + '.%d' % (epoch + 1))
        torch.save(model_rgb.state_dict(), save_path + 'DCF_rgb.pth' + '.%d' % (epoch + 1))
        torch.save(model_depth.state_dict(), save_path + 'DCF_depth.pth' + '.%d' % (epoch + 1))








def val(dataset_path, test_datasets, ckpt_name):
    snapshot = ckpt_name
    is_resnet = opt.is_ResNet
    test_size = opt.trainsize

    if is_resnet:
        model_rgb = DCF_ResNet()
        model_depth = DCF_ResNet()
        model_rgb.load_state_dict(torch.load('./ckpt/DCF_Resnet/'+'DCF_rgb.pth' +snapshot))
        model_depth.load_state_dict(torch.load('./ckpt/DCF_Resnet/' +'DCF_depth.pth' + snapshot))
    else:
        model_rgb = DCF_VGG()
        model_depth = DCF_VGG()
        model_rgb.load_state_dict(torch.load('./ckpt/DCF_VGG/'+'DCF_rgb.pth' + snapshot))
        model_depth.load_state_dict(torch.load('./ckpt/DCF_VGG/' + 'DCF_depth.pth' + snapshot))
    cuda = torch.cuda.is_available()
    if cuda:
        model_rgb.cuda()
        model_depth.cuda()
    model_rgb.eval()
    model_depth.eval()

    for dataset in test_datasets:
        if is_resnet:
            save_path = './results/ResNet50/' + dataset + '/'
        else:
            save_path = './results/VGG16/' + dataset + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_ro = dataset_path + dataset + '/test_images/'
        gt_ro = dataset_path + dataset + '/test_masks/'
        depth_ro = dataset_path + dataset + '/test_depth/'
        test_loader = test_dataset(image_ro, gt_ro, depth_ro, test_size)
        print('Evaluating dataset: %s' %(dataset))

        '''~~~ YOUR FRAMEWORK~~~'''
        for i in trange(test_loader.size):
            image, gt, depth, name = test_loader.load_data()
            depth = torch.cat([depth, depth, depth], dim=1)

            if cuda:
                image = image.cuda()
                depth = depth.cuda()

            _, res_r, x3_r, x4_r, x5_r = model_rgb(image)
            _, res_d, x3_d, x4_d, x5_d = model_depth(depth)
            res = res_r
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            io.imsave(save_path + name, np.uint8(res * 255))
        mae = evalateSOD(save_path, gt_ro, dataset, ckpt_name)
    return mae






def load_two_stream_RANKING(key_min):
    snapshot = key_min
    is_resnet = opt.is_ResNet
    test_size = opt.trainsize

    if is_resnet:
        model_rgb = DCF_ResNet()
        model_depth = DCF_ResNet()
        model_rgb.load_state_dict(torch.load('./ckpt/DCF_Resnet/'+'DCF_rgb.pth' +snapshot))
        model_depth.load_state_dict(torch.load('./ckpt/DCF_Resnet/' +'DCF_depth.pth' + snapshot))
    else:
        model_rgb = DCF_VGG()
        model_depth = DCF_VGG()
        model_rgb.load_state_dict(torch.load('./ckpt/DCF_VGG/'+'DCF_rgb.pth' + snapshot))
        model_depth.load_state_dict(torch.load('./ckpt/DCF_VGG/' + 'DCF_depth.pth' + snapshot))
    cuda = torch.cuda.is_available()
    if cuda:
        model_rgb.cuda()
        model_depth.cuda()
    model_rgb.eval()
    model_depth.eval()


    if is_resnet:
        save_path = './results/ResNet50/temp/'
        image_path_new = './results/ResNet50/temp/train_images/'
        depth_path_new = './results/ResNet50/temp/train_depth/'
        neg_path = './results/ResNet50/temp/train_depth/0/'
        pos_path = './results/ResNet50/temp/train_depth/1/'
    else:
        save_path = './results/VGG16/temp/'
        image_path_new = './results/VGG16/temp/train_images/'
        depth_path_new = './results/VGG16/temp/train_depth/'
        neg_path = './results/VGG16/temp/train_depth/0/'
        pos_path = './results/VGG16/temp/train_depth/1/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        os.makedirs(image_path_new)
        os.makedirs(depth_path_new)
        os.makedirs(neg_path)
        os.makedirs(pos_path)

    image_rt = image_root
    gt_rt = gt_root
    depth_rt = depth_root
    test1_loader = test_dataset(image_rt, gt_rt, depth_rt, test_size)
    print('Generating the new training dataset begin: ')


    IOU_dict = dict()
    '''~~~ YOUR FRAMEWORK~~~'''
    for i in trange(test1_loader.size):
        image, gt, depth, name = test1_loader.load_data()
        img_name = name.split('.png')[0] + '.jpg'
        depth = torch.cat([depth, depth, depth], dim=1)

        if cuda:
            image = image.cuda()
            depth = depth.cuda()

        _, res_r, x3_r, x4_r, x5_r = model_rgb(image)
        _, res_d, x3_d, x4_d, x5_d = model_depth(depth)
        res_r = torch.sigmoid(res_r)
        res_d = torch.sigmoid(res_d)

        # calculating the IoU (GT and depth & RGB)
        res_r_copy = res_r.detach().squeeze().cpu().numpy().astype('int32')
        res_d_copy = res_d.detach().squeeze().cpu().numpy().astype('int32')
        gt_copy = np.array(gt).astype('int32')
        iou_r = iou(res_r_copy,gt_copy)
        iou_d = iou(res_d_copy,gt_copy)

        # Ranking & divide into two sub-sets
        if len(os.listdir(pos_path)) <= 0.2*len(test1_loader):
            if iou_d >= iou_r:
                shutil.copy(image_rt+img_name, image_path_new+img_name)
                shutil.copy(depth_rt+name, pos_path+name)
                IOU_dict[name]=iou_d
            else:
                IOU_dict[name] = iou_d
        else:
            IOU_dict[name] = iou_d


        curr_pos_list = os.listdir(pos_path)

        if i+1 == test1_loader.size:
            # Generating Positive Instance
            if len(os.listdir(pos_path)) <= 0.2*len(test1_loader):
                sort_iou = sorted(IOU_dict.items(), key =lambda item:item[1], reverse=True)
                for item_i in sort_iou:
                    if item_i[0] in curr_pos_list:
                        pass
                    elif len(os.listdir(pos_path)) <= 0.2 * len(test1_loader):
                        im_name = item_i[0].split('.png')[0] + '.jpg'
                        shutil.copy(image_rt + im_name, image_path_new + im_name)
                        shutil.copy(depth_rt + item_i[0], pos_path + item_i[0])
                    else:
                        pass
            else:
                pass

            # Generating Negative Instance
            sort_iou_rev = sorted(IOU_dict.items(), key =lambda item:item[1])
            if len(os.listdir(neg_path)) <= 0.2 * len(test1_loader):
                for item_j in sort_iou_rev:
                    if len(os.listdir(neg_path)) <= 0.2 * len(test1_loader):
                        im_name = item_j[0].split('.png')[0] + '.jpg'
                        shutil.copy(image_rt + im_name, image_path_new + im_name)
                        shutil.copy(depth_rt + item_j[0], neg_path + item_j[0])
                    else:
                        pass

            '''~~~Prevent overfitting~~~'''
            # # Deleting the repeating files
            # neg_list = os.listdir(neg_path)
            # pos_list = os.listdir(pos_path)
            # common_list = [x for x in pos_list if x in neg_list]
            # for del_i in common_list:
            #     os.remove(neg_path+del_i)
            #     os.remove(pos_path+del_i)
            #     os.remove(image_path_new + del_i.split('.png')[0] + '.jpg')

    return image_path_new, pos_path, neg_path




print("Let's go!")
MAE_index = dict()
for epoch in range(1, opt.epoch):
    if epoch <= opt.calib_flag -1:
        adjust_lr(optimizer_rgb, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        adjust_lr(optimizer_depth, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train(train_loader, model_rgb, model_depth,
            optimizer_rgb, optimizer_depth, epoch)
        if (epoch+1) % 5 == 0:
            ckpt_name = '.' + str(epoch+1)
            mae = val(val_root, validation,ckpt_name)
            MAE_index[ckpt_name] = mae

    elif epoch == opt.calib_flag:
        key_min = min(MAE_index.keys(), key=(lambda k: MAE_index[k]))
        print('The first training phase is over, loading the best ckpt: %s' %(str(key_min)))
        image_path_new, pos_path, neg_path= load_two_stream_RANKING(key_min)

        # calibration_loader = calibration_dataset(image_path_new, pos_path, neg_path, opt.trainsize)
        calibration_loader = get_calibration_loader(image_path_new, pos_path, neg_path, batchsize=opt.batchsize,
                                                    testsize=opt.trainsize,shuffle=True,num_workers=12,pin_memory=True)
        total_step_cal = len(calibration_loader)
        adjust_lr(optimizer_dis, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        adjust_lr(optimizer_estimator, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train_calibration(calibration_loader, model_rgb,model_depth, model_discriminator, model_estimator,
                          optimizer_dis, optimizer_estimator, epoch, key_min)

    elif epoch < opt.epoch -1:
        adjust_lr(optimizer_dis, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        adjust_lr(optimizer_estimator, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        #calibration_loader = calibration_dataset(image_path_new, pos_path, neg_path, opt.trainsize)
        calibration_loader = get_calibration_loader(image_path_new, pos_path, neg_path, batchsize=opt.batchsize,
                                                    testsize=opt.trainsize, shuffle=True, num_workers=12,
                                                    pin_memory=True)
        train_calibration(calibration_loader, model_rgb,model_depth, model_discriminator, model_estimator,
                          optimizer_dis, optimizer_estimator, epoch, key_min)
    elif epoch >= opt.epoch -1:
        writer.close()
