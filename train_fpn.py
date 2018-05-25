import data_loader
import model
import fpn_res18 as fpn
import torch 
import torch.nn as nn
import torch.backends.cudnn as cudnn
import cv2
import argparse
import torchvision.transforms as transforms
import time
import os
import numpy as np

def parse():
    parser = argparse.ArgumentParser()
    return parser.parse_args()

def construct_model():
    net = fpn.pose_estimation()
#    net.load_weights(model_path='models/model_iter_80000.pkl')
    net.cuda()
    return net

def main():
    anno_path='/data/xiaobing.wang/pingjun.li/yfji/hand_labels_synth/hand_label_cropt.json'
    batch=8
    base_lr=0.0001
    decay_ratio=0.125
    max_iters=120000
    stepvalues=[72000, 100000, 120000]
    g_steps=stepvalues[0]
    
    display=20
    snapshot=20000
    
    cudnn.benchmark = True
    train_loader = torch.utils.data.DataLoader(
		data_loader.DataLoader(anno_path, transforms=transforms.ToTensor(), batch_size=batch),
		batch_size=batch, shuffle=True,
		num_workers=6, pin_memory=True)
    
    net=construct_model()
    
    criterion = nn.MSELoss().cuda()
    params = []
    for key, value in net.named_parameters():
#        print(key,value.size())
        if value.requires_grad != False:
            print(key,value.shape)
            params.append({'params': value, 'lr': base_lr})

   
    optimizer = torch.optim.SGD(params, base_lr, momentum=0.9,
	                            weight_decay=0.0005)
    
    iters=0
    lr=base_lr
    step=0
    step_index=0
    
    heat_weight=32*32*22*0.5
    
    while iters<max_iters:
        for i, (data, label) in enumerate(train_loader):
            data_var=torch.autograd.Variable(data.cuda(async=True))
            label_var=torch.autograd.Variable(label.cuda(async=True))
            heatmap=net(data_var)
            loss=criterion(heatmap, label_var)*heat_weight

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
   
            rate=lr*np.power(decay_ratio, 1.0*step/g_steps)
            for param_group in optimizer.param_groups:
                param_group['lr']=rate

            if iters%display==0:
                print('[%d/%d] loss: %f, learn rate: %e'%(iters, max_iters, loss, rate))
            if iters==stepvalues[step_index]:
                print('learning rate decay: %e'%rate)
                step=0
                lr=rate
                g_steps=stepvalues[step_index+1]-stepvalues[step_index]
                step_index+=1
            if iters>0 and iters%snapshot==0:
                model_name='models_fpn/model_iter_%d.pkl'%iters
                print('Snapshotting to %s'%model_name)
                torch.save(net.state_dict(),model_name)
            step+=1
            iters+=1
            if iters==max_iters:
                break
    model_name='models_fpn/model_iter_%d.pkl'%max_iters
    print('Snapshotting to %s'%model_name)
    torch.save(net.state_dict(),model_name)

if __name__=='__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    main()
    
