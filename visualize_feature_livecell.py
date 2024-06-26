import matplotlib as mpl
# we cannot use remote server's GUI, so set this  
mpl.use('Agg')
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import models
from modeling import *
import os
from os.path import join
import matplotlib.pyplot as plt
from matplotlib import cm as CM
import argparse
import shutil

import json
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def preprocess_image(cv2im, resize_im=False):
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
    # Resize image
    if resize_im:
        cv2im = cv2.resize(cv2im, (224, 224))
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
 
 
class FeatureVisualization():
    def __init__(self,img_path,model_path,selected_layer):
        self.img_path = img_path
        self.model_path=model_path
        self.selected_layer = selected_layer
        self.pretrained_model = MARNet(load_model=model_path, downsample=1, objective='dmp+amp', save_feature=True)
 
    def process_image(self):
        img = cv2.imread(self.img_path)
        img = preprocess_image(img)
        self.img = img
        return img
        
    def get_single_feature(self, num):
        input_img = self.process_image()
        print('input_img.shape:', input_img.shape)
        x=input_img
        outputs = self.pretrained_model(x)
        amp4, amp3, amp2, amp1, amp0 = outputs[-5:]
        fea_d = dict()
        amp_d = dict()
        amp_d['amp4'] = amp4
        amp_d['amp3'] = amp3
        amp_d['amp2'] = amp2
        amp_d['amp1'] = amp1
        amp_d['amp0'] = amp0
        
        fea_d['xb4_before'] = self.pretrained_model.xb4_before[:,0:num,:,:].squeeze_()
        fea_d['xb4_after'] = self.pretrained_model.xb4_after[:,0:num,:,:].squeeze_()
        
        fea_d['xb3_before'] = self.pretrained_model.xb3_before[:,0:num,:,:].squeeze_()
        fea_d['xb3_after'] = self.pretrained_model.xb3_after[:,0:num,:,:].squeeze_()
        
        fea_d['xb2_before'] = self.pretrained_model.xb2_before[:,0:num,:,:].squeeze_()
        fea_d['xb2_after'] = self.pretrained_model.xb2_after[:,0:num,:,:].squeeze_()
        
        fea_d['xb1_before'] = self.pretrained_model.xb1_before[:,0:num,:,:].squeeze_()
        fea_d['xb1_after'] = self.pretrained_model.xb1_after[:,0:num,:,:].squeeze_()
        
        fea_d['xb0_before'] = self.pretrained_model.xb0_before[:,0:num,:,:].squeeze_()
        fea_d['xb0_after'] = self.pretrained_model.xb0_after[:,0:num,:,:].squeeze_()
        
        return amp_d, fea_d
 
    def save_feature_to_img(self):
        save_dir='/content/drive/MyDrive/MARUNet/figs/livecell/visual_feature'
        image_name = os.path.basename(self.img_path)
        fol_name=image_name.split('.')[0]
        save_path=join(save_dir,fol_name)
        print('save path is ',save_path)
        try:
          os.makedirs(save_path, exist_ok=True)
          if os.path.exists(save_path):
              print("Directory created:", save_path)
          else:
              print("Directory creation failed:", save_path)
        except Exception as e:
            print("Error creating directory:", e)

        with open('/content/drive/MyDrive/MARUNet/live_cell/train_skov3.json', 'r') as f:
          data = json.load(f)
        image_filename=image_name
        image_id = None
        for img_data in data['images']:
            if img_data['file_name'] == image_filename:
                image_id = img_data['id']
                break
        segmentations = []
        if image_id is not None:
            for ann_data in data['annotations']:
                if ann_data['image_id'] == image_id:
                    segmentations.append(ann_data['segmentation'])
        image = cv2.imread(self.img_path)
        # Plot the image
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for segmentation in segmentations:
          # Each segmentation is a list of lists
          for seg in segmentation:
              poly = np.array(seg).reshape((len(seg) // 2, 2))
              polygon = Polygon(poly, linewidth=1, edgecolor='r', facecolor='none')
              plt.gca().add_patch(polygon)
        plt.title(image_filename)
        plt.axis('off')
        save_filename = image_filename.split(".")[0] + "_with_contours.jpg"
        save_filepath = os.path.join(save_path, save_filename)
        plt.savefig(save_filepath)
        plt.close()

        #copy original image as refernce 
        # shutil.copy(self.img_path,join(save_path,image_name))
        
        #to numpy
        num = 4
        amps, features = self.get_single_feature(num)
        height, width = self.img.shape[2:]
        for item in amps.items():
            k,v = item
            v = v.squeeze_().detach().numpy() 
            fig, ax = plt.subplots()
            ax.imshow(v, cmap=CM.jet)
            fig.set_size_inches(width/400.0, height/400.0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.axis('off')
            plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
            plt.margins(0,0)
            plt.savefig(join(save_path,'sha130_{}.jpg'.format(k)), dpi=300)
            np.save(os.path.join(save_path, 'sha130_{}.npy'.format(k)), v)
            plt.clf()
            
        for item in features.items():
            k, v = item
            
            col = 2
            row = num // col
            
            plt.figure(figsize=(width*col/400.0,height*row/400.0))
            for i in range(row):
                for j in range(col):
                    feature=v[i * col + j].detach().numpy() 
                    #feature = v[i * col + j].data.numpy()
                    plt.subplot(row, col, i * col + j + 1)
                    plt.imshow(feature, cmap=CM.jet)
                    plt.axis('off')
            plt.subplots_adjust(top=0.99,bottom=0.01,left=0.01,right=0.99,hspace=0.01,wspace=0.01)
            plt.margins(0,0)
            
            plt.savefig(join(save_path,'sha130_{}.jpg'.format(k)), dpi=300)
            np.save(os.path.join(save_path, 'sha130_{}.npy'.format(k)), feature)
            plt.clf()

def main(args):
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
  if args.img_path != '':
    myClass=FeatureVisualization(args.img_path,args.model_path,5)
  #print (myClass.pretrained_model)

    myClass.save_feature_to_img()
  else:
    print('please input image path using --img_path')
if __name__=='__main__':
    # get class
    #ahsan
    parser = argparse.ArgumentParser(description='visualize the image')
    parser.add_argument('--img_path', metavar='image_path', default='', type=str)
    parser.add_argument('--model_path', metavar='model_path', default='', type=str)

    args = parser.parse_args()
    main(args)
    
