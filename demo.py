import model
import fpn_res18 as fpn
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import cv2
import argparse
import torchvision.transforms as transforms
import os
import os.path as op
import util
import numpy as np


def run_model(image_name, threshold, net):
    image=cv2.imread(image_name)
    input_side=256
    scale=1.0*input_side/max(image.shape[0],image.shape[1])
    image=cv2.resize(image, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    pad_image=np.zeros((input_side,input_side,3),dtype=np.float32)
    pad_image+=128
    pad_image[:image.shape[0],:image.shape[1],:]=image.astype(np.float32)
    
    data=pad_image.transpose(2,0,1)
    data=(data-128)/255.0
    data=torch.Tensor(data[np.newaxis,:,:,:])
    
    output=net(torch.autograd.Variable(data.cuda())).data.cpu().numpy().squeeze()
    stride=8
    output=cv2.resize(output.transpose(1,2,0), (0,0), fx=1.0*stride,fy=1.0*stride, interpolation=cv2.INTER_CUBIC)
    #print(output.shape)
    heatmap=output[:,:,21]

    heatmap_bgr=np.zeros(image.shape, dtype=np.uint8)
    for i in range(heatmap_bgr.shape[0]):
        for j in range(heatmap_bgr.shape[1]):
            heatmap_bgr[i,j,[2,1,0]]=util.getJetColor(1-heatmap[i,j],0,1)
            
#    print(heatmap_bgr.shape)
    out_image=cv2.addWeighted(image, 0.7, heatmap_bgr, 0.3, 0).astype(np.uint8)
    
    connections=[[0,1],[1,2],[2,3],[3,4],
          [0,5],[5,6],[6,7],[7,8],
          [0,9],[9,10],[10,11],[11,12],
          [0,13],[13,14],[14,15],[15,16],
          [0,17],[17,18],[18,19],[19,20]]
          
    for i in range(output.shape[2]-1):
        h_map=output[:,:,i]
        peaks=util.find_peak(h_map, thresh=threshold)
        for pt in peaks:
            cv2.circle(image, (pt[0],pt[1]), 3, (0,255,255), -1)
            
    for conn in connections:
        start=conn[0]
        end=conn[1]
        peak_start=util.find_peak(output[:,:,start], thresh=threshold)
        peak_end=util.find_peak(output[:,:,end], thresh=threshold)
        if len(peak_start)==0 or len(peak_end)==0:
            continue
        peak_start=sorted(peak_start, key=lambda x:x[2], reverse=True)
        peak_end=sorted(peak_end, key=lambda x:x[2], reverse=True)
        cv2.line(image, (peak_start[0][0],peak_start[0][1]),(peak_end[0][0],peak_end[0][1]), (0,0,255), 2)
    return out_image, image
    
def main(parser):
    args=parser.parse_args()
    threshold=args.threshold
    filename=args.filename
    imagedir=args.imagedir
    savedir=args.savedir
    net_type=args.name
    
    net=None
    if net_type=='res18':
        net=fpn.pose_estimation(pretrain=True)
        model_path='models_fpn/model_iter_20000.pkl'
        net.load_weights(model_path=model_path)
    elif net_type=='mobilenet':
        net=model.Mobilenet(pretrain=True)
        model_path='models/model_iter_40000.pkl'
        net.load_weights(model_path=model_path)
    else:
        raise Exception('Unknown network architecture')
        
    net.cuda()
    
    if filename!='' and imagedir!='':
        raise Exception('Only one of image_dir and image can be used')
    if imagedir!='':
        if imagedir[0]!='/':
            imagedir=op.join(os.getcwd(),imagedir)
        if savedir=='':
            savedir=op.join(os.getcwd(),'preds')
    if savedir[0]!='/':
        savedir=op.join(os.getcwd(),savedir)
        imagenames=os.listdir(imagedir)
        num_samples=len(imagenames)
        cnt=0
        for imagename in imagenames:
            print('[%d/%d]'%(cnt+1,num_samples))
            imagename_no_format=imagename[:imagename.rfind('.')]
            imagename=op.join(imagedir, imagename)
            heatmap, skeletonmap=run_model(imagename,threshold, net)
            cv2.imwrite(op.join(savedir,'%s_heatmap.jpg'%imagename_no_format),heatmap)
            cv2.imwrite(op.join(savedir,'%s.jpg'%imagename_no_format), skeletonmap)
            cnt+=1
    print('done')
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net', dest='name', type=str, default='res18',help='mobilenet or resnet18')
    parser.add_argument('--thresh', dest='threshold', type=float,default=0.2, help='threshold for heatmap')
    parser.add_argument('--image', dest='filename', type=str, default='', help='image file path')
    parser.add_argument('--imagedir', dest='imagedir', type=str, default='', help='directory containing images')
    parser.add_argument('--savedir', dest='savedir', type=str, default='', help='directory to save predicted images')
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main(parser)