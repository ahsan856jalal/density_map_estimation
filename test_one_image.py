import cv2
import numpy as np
import torch
from torch.autograd import Variable
from modeling import *
import os
from matplotlib import cm as CM
import matplotlib.pyplot as plt
from PIL import Image
import h5py
import argparse
# set your pretrained model path here
model_paths = {'sha':{'MARNet':"/content/drive/MyDrive/MARUNet/MARNet_sha.pth",
                              'U_VGG':"/home/datamining/Models/CrowdCounting/U_VGG_d1_sha_random_cr0.5_ap3avg-ms-ssim_v50_bg_lsn6.pth",
                              'CSRNet':"/home/datamining/Models/CrowdCounting/PartAmodel_best.pth.tar"},
                       'shb':{'MARNet':"/home/datamining/Models/CrowdCounting/MARNet_d1_shb_random_cr0.5_3avg-ms-ssim_v50_amp0.15_bg_lsn6.pth",
                              'U_VGG':"/home/datamining/Models/CrowdCounting/U_VGG_d1_shb_random_cr0.5_3avg-ms-ssim_v50_bg_lsn6.pth",
                              'CSRNet':"/home/datamining/Models/CrowdCounting/partBmodel_best.pth.tar"},
                       'qnrf':{'MARNet':"/home/datamining/Models/CrowdCounting/MARNet_d1_qnrf_random_cr0.5_3avg-ms-ssim_v50_amp0.16_bg_lsn6.pth"},
                               'U_VGG':"/home/datamining/Models/CrowdCounting/U_VGG_d1_qnrf_random_cr0.5_3avg-ms-ssim_v50_bg_lsn6.pth",
                               }
def test_image(image_path):
    # Load the image
    img = cv2.imread(image_path)
    
    # Check if the image is loaded correctly
    if img is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")
    
    # Convert image to contiguous array and change from BGR to RGB
    im_as_arr = np.ascontiguousarray(img[..., ::-1])
    
    return im_as_arr
def preprocess_image(cv2im):
    """
        Processes image for CNNs
    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (Pytorch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    im_as_arr = np.float32(cv2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var

def load_model(models, model_paths, dataset='sha'):
    pretrained_models = {}
    for model in models:
        if model == 'MARNet':
            pretrained_model = MARNet(load_model=model_paths, downsample=1, objective='dmp+amp')
        elif model == 'MSUNet':
            pretrained_model = U_VGG(load_model=model_paths[dataset]['U_VGG'], downsample=1)
        elif model == 'CSRNet':
            pretrained_model = CSRNet(load_model=model_paths[dataset][model], downsample=8)
        pretrained_models[model]=pretrained_model
    return pretrained_models
    
def map_normalize(image):
    return ((image - image.min()) / (image.max() - image.min()))
    
def img_test(pretrained_model, img_path, divide=50, ds=1):
    img = cv2.imread(img_path)
    img=test_image(img_path)
    img = preprocess_image(img)
    if torch.cuda.is_available():
        img = img
    outputs = pretrained_model(img)
    
    if torch.cuda.is_available():
        dmp = outputs[0].squeeze().detach().cpu().numpy()
        amp = outputs[-1].squeeze().detach().cpu().numpy()
        file_name1 = os.path.basename(args.img_path)
        file_name, file_extension = os.path.splitext(file_name1)
        model_name1=os.path.basename(args.model_path)
        model_name, _ = os.path.splitext(model_name1)
        saving_dir=os.path.join(args.save_path,file_name,model_name)
        os.makedirs(saving_dir,exist_ok=True)
        dmp_file_path = os.path.join(saving_dir,'dmp_data.h5')
        amp_file_path = os.path.join(saving_dir,'amp_data.h5')

        # Save the dmp and amp arrays as .h5 files
        with h5py.File(dmp_file_path, 'w') as dmp_file:
            dmp_file.create_dataset('density', data=dmp)

        with h5py.File(amp_file_path, 'w') as amp_file:
            amp_file.create_dataset('amp', data=amp)

        dmp_normalized = (map_normalize(dmp) * 255).astype(np.uint8)
        amp_normalized = (map_normalize(amp) * 255).astype(np.uint8)
        dmp_normalized = cv2.applyColorMap((dmp_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        amp_normalized = cv2.applyColorMap((amp_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(saving_dir,"dmp.png"), dmp_normalized)
        cv2.imwrite(os.path.join(saving_dir,"amp.png"), amp_normalized)
        # dmp_img = Image.fromarray(dmp_normalized)
        # amp_img = Image.fromarray(amp_normalized)
        # dmp_img.save(os.path.join(saving_dir,"dmp.png"))
        # amp_img.save(os.path.join(saving_dir,"amp.png"))
    else:
        dmp = outputs[0].squeeze().detach().numpy()
        amp = outputs[-1].squeeze().detach().numpy()
    dmp = dmp/divide
    print('estimated cell count: ', dmp.sum())
    return dmp
def main(args):
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  if args.img_path != '' and args.model_path != '':
    model = load_model(['MARNet'],args.model_path)
    if torch.cuda.is_available():
        print('cuda available')

        model = model
    else:
      print('no cuda')
    dmp = img_test(model['MARNet'], args.img_path, divide=50, ds=1)

  else:
    print('please input image path using --img_path and --model_path for model' )
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visualize the image')
    parser.add_argument('--img_path', metavar='image_path', default='', type=str)
    parser.add_argument('--model_path', metavar='model_path', default='', type=str)
    parser.add_argument('--save_path', metavar='save_path', default='', type=str)

    args = parser.parse_args()
    main(args)


