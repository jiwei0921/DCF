import torch
import torch.nn.functional as F

import numpy as np
import pdb, os, argparse
from skimage import io
from tqdm import trange
from model.DCF_models import DCF_VGG
from model.DCF_ResNet_models import DCF_ResNet
from model.fusion import fusion
from evaluateSOD.main import evalateSOD
from data import test_dataset
from model.depth_calibration_models import discriminator, depth_estimator


def eval_data(dataset_path, test_datasets, ckpt_name):

    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--is_ResNet', type=bool, default=True, help='VGG or ResNet backbone')
    parser.add_argument('--snapshot', type=str, default=ckpt_name, help='checkpoint name')
    cfg = parser.parse_args()

    if cfg.is_ResNet:
        model_rgb = DCF_ResNet()
        model_depth = DCF_ResNet()
        model = fusion()
        model_discriminator = discriminator(n_class=2)
        model_estimator = depth_estimator()
        model_rgb.load_state_dict(torch.load('./ckpt/DCF_Resnet/'+'DCF_rgb.pth' +cfg.snapshot))
        model_depth.load_state_dict(torch.load('./ckpt/DCF_Resnet/' +'DCF_depth.pth' + cfg.snapshot))
        model.load_state_dict(torch.load('./ckpt/DCF_Resnet/' + 'DCF.pth' +cfg.snapshot))
        model_discriminator.load_state_dict(torch.load('./ckpt/DCF_Resnet/' + 'DCF_dis.pth' + cfg.snapshot))
        model_estimator.load_state_dict(torch.load('./ckpt/DCF_Resnet/' + 'DCF_estimator.pth' + cfg.snapshot))
    else:
        model_rgb = DCF_VGG()
        model_depth = DCF_VGG()
        model = fusion()
        model_discriminator = discriminator(n_class=2)
        model_estimator = depth_estimator()
        model_rgb.load_state_dict(torch.load('./ckpt/DCF_VGG/'+'DCF_rgb.pth' +cfg.snapshot))
        model_depth.load_state_dict(torch.load('./ckpt/DCF_VGG/' + 'DCF_depth.pth' +cfg.snapshot))
        model.load_state_dict(torch.load('./ckpt/DCF_VGG/' +'DCF.pth' + cfg.snapshot))
        model_discriminator.load_state_dict(torch.load('./ckpt/DCF_VGG/' + 'DCF_dis.pth' + cfg.snapshot))
        model_estimator.load_state_dict(torch.load('./ckpt/DCF_Resnet/' + 'DCF_estimator.pth' + cfg.snapshot))


    cuda = torch.cuda.is_available()
    if cuda:
        model_rgb.cuda()
        model_depth.cuda()
        model.cuda()
        model_discriminator.cuda()
        model_estimator.cuda()
    model_rgb.eval()
    model_depth.eval()
    model.eval()
    model_estimator.eval()
    model_discriminator.eval()


    for dataset in test_datasets:
        if cfg.is_ResNet:
            save_path = './results/ResNet50/' + dataset + '/'
        else:
            save_path = './results/VGG16/' + dataset + '/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_root = dataset_path + dataset + '/test_images/'
        gt_root = dataset_path + dataset + '/test_masks/'
        depth_root = dataset_path + dataset + '/test_depth/'
        test_loader = test_dataset(image_root, gt_root, depth_root, cfg.testsize)
        print('Evaluating dataset: %s' %(dataset))


        '''~~~ YOUR FRAMEWORK~~~'''
        for i in trange(test_loader.size):
            image, gt, depth, name = test_loader.load_data()

            if cuda:
                image = image.cuda()
                depth = depth.cuda()

            # RGB Stream
            _, res_r,x3_r,x4_r,x5_r = model_rgb(image)

            # depth calibration
            score= model_discriminator(depth)
            score = torch.softmax(score,dim=1)
            x3_, x4_, x5_ = x3_r.detach(), x4_r.detach(), x5_r.detach()
            pred_depth = model_estimator(image,x3_, x4_, x5_)
            depth_calibrated = torch.mul(depth, score[:, 0].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).
                                         expand(-1, 1, cfg.testsize, cfg.testsize)) \
                               + torch.mul(pred_depth, score[:, 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).
                                           expand(-1, 1, cfg.testsize, cfg.testsize))
            depth_calibrated = torch.cat([depth_calibrated, depth_calibrated, depth_calibrated], dim=1)

            # Depth Stream
            _, res_d,x3_d,x4_d,x5_d = model_depth(depth_calibrated)
            # Fusion Stream (CRM)
            _,res,_,_,_ = model(x3_r,x4_r,x5_r,x3_d,x4_d,x5_d)


            res = res+res_d+res_r
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            io.imsave(save_path+name, np.uint8(res * 255))
        _ = evalateSOD(save_path, gt_root, dataset,ckpt_name)
    return

if __name__ == '__main__':

    dataset_path = '/Data/test_data/'
    test_datasets=['LFSD']
    # test_datasets = ['NJU2K', 'NLPR', 'SIP', 'LFSD', 'DES','SSD','DUT','STERE']

    ckpt_name = '.85'

    eval_data(dataset_path,test_datasets,ckpt_name)