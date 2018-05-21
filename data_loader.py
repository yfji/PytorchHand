import torch
import torch.utils.data as data
import json
import data_augment as da
import numpy as np
import cv2
import os.path as op

class DataLoader(data.Dataset):
    def __init__(self, anno_path=None, log_file=None, transforms=None, batch_size=8):
        self.anno_path=anno_path
        self.transforms=transforms
        
        label=json.load(open(self.anno_path, 'r'))
        self.dataset=label['root']
        self.num_samples=len(self.dataset)
        
        self.batch_size=batch_size
        self.center_perterb_max=20
        self.angle_max=25
        self.target_scale=0.8
        self.scale_range=[0.7,1.2]
        self.stride=8
        self.num_parts=21
        self.label_channels=22
        self.sigma=6.5
        self.visualize=True
        self.savedir='./visualize'
        self.image_index=0
        self.net_input_size=256
        
        self.log_file=log_file
        self.shuffle()
        
    def __getitem__(self, index):
        image, keypoints, img_path=self.parse_anno(index)
        
        base_scale=1.0*self.target_scale/(1.0*image.shape[0]/self.net_input_size)
        
        label_side=int(self.net_input_size/self.stride)
        imagelabel=np.zeros((label_side,label_side,self.label_channels),dtype=np.float32)

        labeled_index=(keypoints[:,-1]!=2)
        if len(np.nonzero(labeled_index)[0])>0:
            image=da.aug_scale(image, base_scale, self.scale_range, keypoints)
            image=da.aug_rotate(image, self.angle_max, keypoints)
            center=np.mean(keypoints[labeled_index,:2],axis=0) 
            image,flag=da.aug_crop(image, center, self.net_input_size, self.center_perterb_max, keypoints)
        
            self.putGaussianMap(imagelabel, keypoints, sigma=self.sigma)
        
            if flag==0:
                g_map=imagelabel[:,:,-1]
                g_map=cv2.resize(g_map, (0,0), fx=self.stride,fy=self.stride,interpolation=cv2.INTER_CUBIC)
                raw_image=image.astype(np.uint8)
                vis_img=self.add_weight(raw_image,g_map)
                cv2.imwrite('0.jpg', vis_img)
                assert(0)
                            
            if self.visualize and self.image_index<50:
                g_map=imagelabel[:,:,-1]
                g_map=cv2.resize(g_map, (0,0), fx=self.stride,fy=self.stride,interpolation=cv2.INTER_CUBIC)
                raw_image=image.astype(np.uint8)
                vis_img=self.add_weight(raw_image,g_map)
                cv2.imwrite(op.join(self.savedir,'sample_%d.jpg'%self.image_index),vis_img)
                self.image_index+=1
        else:
            c_image=128*np.ones((self.net_input_size,self.net_input_size,3),dtype=np.float32)
            h=min(c_image.shape[0],image.shape[0])
            w=min(c_image.shape[1],image.shape[1])
            c_image[:h,:w,:]=image[:h,:w,:]
            image=c_image
            self.log_file.write(img_path+'\n')
        image-=128.0
        image/=255.0
        return torch.from_numpy(image.transpose(2,0,1)), torch.from_numpy(imagelabel.transpose(2,0,1))

    def __len__(self):
        return self.num_samples
    
    def shuffle(self):
        self.random_order=np.random.permutation(np.arange(self.num_samples))
        self.cur_index=0
        
    def parse_anno(self, index):
        entry=self.dataset[self.random_order[index]]
        img_path=entry['img_paths']
        crop_x=int(entry['crop_x'])
        crop_y=int(entry['crop_y'])
        img_height=int(entry['img_height'])
        img_width=int(entry['img_width'])
        image=cv2.imread(img_path)
        image=image[crop_y:crop_y+img_height,crop_x:crop_x+img_width,:]

#        hand_pos=entry['objpos']
        keypoints=np.asarray(entry['joint_self'])   #21x3
        x_outs=np.logical_or(keypoints[:,0]<0,keypoints[:,0]>image.shape[1])
        y_outs=np.logical_or(keypoints[:,1]<0,keypoints[:,1]>image.shape[0])
        #avoid kpts coordinates out of image
        invalid=np.logical_or(x_outs,y_outs)
        keypoints[invalid]=np.asarray([0.,0.,2.])
        return image, keypoints, img_path
    
    def putGaussianMap(self, label, keypoints, sigma=7.0):
        start = self.stride / 2.0 - 0.5
        for i in range(label.shape[2]-1):    #[h,w,c]
            kp=keypoints[i]
            if kp[-1]!=2:
                for y in range(label.shape[0]):
                    for x in range(label.shape[1]):
                        yy = start + y * self.stride
                        xx = start + x * self.stride
                        dis = ((xx - kp[0]) * (xx - kp[0]) + (yy - kp[1]) * (yy - kp[1])) / 2.0 / sigma / sigma
                        if dis > 4.6052:
                            continue
                        label[y,x,i] += np.exp(-dis)
                        label[y,x,i]=min(1,label[y,x,i])
        label[:,:,-1]=np.max(label[:,:,:-1],axis=2)
        
    def add_weight(self,image, g_map):
        heatmap_bgr=np.zeros(image.shape, dtype=np.uint8)
        for i in range(heatmap_bgr.shape[0]):
            for j in range(heatmap_bgr.shape[1]):
                heatmap_bgr[i,j,[2,1,0]]=self.getJetColor(1-g_map[i,j],0,1)
        out_image=cv2.addWeighted(image, 0.7, heatmap_bgr, 0.3, 0).astype(np.uint8)
        return out_image
    
    def getJetColor(self, v, vmin, vmax):
        c = np.zeros((3))
        if (v < vmin):
            v = vmin
        if (v > vmax):
            v = vmax
        dv = vmax - vmin
        if (v < (vmin + 0.125 * dv)): 
            c[0] = 256 * (0.5 + (v * 4)) #B: 0.5 ~ 1
        elif (v < (vmin + 0.375 * dv)):
            c[0] = 255
            c[1] = 256 * (v - 0.125) * 4 #G: 0 ~ 1
        elif (v < (vmin + 0.625 * dv)):
            c[0] = 256 * (-4 * v + 2.5)  #B: 1 ~ 0
            c[1] = 255
            c[2] = 256 * (4 * (v - 0.375)) #R: 0 ~ 1
        elif (v < (vmin + 0.875 * dv)):
            c[1] = 256 * (-4 * v + 3.5)  #G: 1 ~ 0
            c[2] = 255
        else:
            c[2] = 256 * (-4 * v + 4.5) #R: 1 ~ 0.5                      
        return c    

