from email.mime import base
import os
from cv2 import norm
import torch
import argparse
import importlib
from torch.backends import cudnn
cudnn.enabled = True
import imp
from pdb import set_trace
import numpy as np
import torch
from torch.backends import cudnn
cudnn.enabled = True
from torch.utils.data import DataLoader
from PIL import Image
import torch.nn.functional as F
import os.path
import cv2
from torchvision import transforms
from h_channel import *
import math

class InferDataset():
    def __init__(self, image_file,data_points,transform,width,highth):
        img=cv2.imread(image_file)
        self.image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.data_points = data_points
        self.transform=transform
        self.width=width
        self.highth=highth

    def __getitem__(self, index):
        [x_center,y_center]=self.data_points[index]
        img=self.image[x_center-self.width//2:x_center+self.width//2,y_center-self.highth//2:y_center+self.highth//2,:]
        img=self.transform(img)
        return  x_center,y_center,img
        
    def __len__(self):
        return len(self.data_points)

def test_phase(args):

    weights=args.weights #"checkpoint/best.pth"
    thor=args.thor
    input_path=args.input_path #"path"
    output_path=args.output_path #"output_path"

    num_workers=args.num_workers
    batch_size=args.batch_size

    image_path=input_path+"/png"
    label_gt_path=input_path+"/csv"
    savepath=output_path+'/pre'
    txt_output=output_path+"/pre_point"
    pre_h_mask=output_path+"/pre_h_mask"

    for dir_name in [savepath,txt_output,pre_h_mask]:
        if os.path.exists(dir_name):  
            continue                      
        else: 
            os.mkdir(dir_name)  

    width=80
    highth=80

    model = getattr(importlib.import_module("network.resnet38_cls_dassl_multi_label"), 'Net')(1)
    model.load_state_dict(torch.load(weights), strict=False)
    model.eval()
    model.cuda()

    patch_names_list= os.listdir(image_path)

    transform = transforms.Compose([transforms.ToTensor()]) 
    for patch_name in tqdm(patch_names_list):
        infer_points_list=[]
        basename=patch_name[:-4]
        image_file=image_path+'/'+patch_name
        label_file=label_gt_path+'/'+basename+'.csv'
        img = cv2.imread(image_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        [w,h] = img.shape[:2]
        output_map=np.zeros((w,h))
        prob_map=np.zeros((w,h))
        img_temp=cv2.pyrDown(img)
        img_temp = cv2.GaussianBlur(img_temp,(3,3),0) 
        w_channel,h_channel=stain_separate(img_temp)
        h_channel=h_channel[0,:]
        h_channel=h_channel.reshape(math.ceil(w/2),math.ceil(h/2))
        h_channel=h_channel>np.max(h_channel)*0.4
        h_channel=morphology.remove_small_objects(h_channel,min_size=15,connectivity=1)
        label_image =measure.label(h_channel)
        temp=Image.fromarray((h_channel*255).astype('uint8'))
        temp.save(pre_h_mask+"/"+patch_name)

        regions=measure.regionprops(label_image)
        for region in regions:
            x_center=(region.bbox[0]+region.bbox[2])
            y_center=(region.bbox[1]+region.bbox[3])

            if x_center-width//2<0 :
                x_center=width//2
            if y_center-highth//2<0 :
                y_center=highth//2
            if x_center+width//2>w :
                x_center=w-width//2
            if y_center+highth//2>h:
                y_center=h-highth//2

            point=[x_center,y_center]
            infer_points_list.append(point)

        infer_dataset = InferDataset(image_file,infer_points_list,transform=transform,width=width,highth=highth)
        infer_data_loader = DataLoader(infer_dataset,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=False,
                                    batch_size=batch_size
                                    )
        torch.cuda.empty_cache()
        
        for iter,  (x_center,y_center,img_list) in enumerate(infer_data_loader):   
            output2, _, _,_,_ = model(img_list.cuda(),1) 
            prob2 = torch.softmax(output2,dim=-1).cpu().data.numpy()
            for (x,y,img_,pre) in zip(x_center,y_center,img_list,prob2): 
                if pre[0]>thor:  
                    img_=img_.unsqueeze(0)
                    g_cam=model.forward_cam(img_.cuda())
                    g_cam=g_cam.squeeze(0)[0]
                    if x-width//2<0 :
                        x=width//2
                    if y-highth//2<0 :
                        y=highth//2
                    if x+width//2>w:
                        x=w-width//2
                    if y+highth//2>h:
                        y=h-highth//2

                    norm_cam = np.array(g_cam.cpu().detach())
                    _range = np.max(norm_cam) - np.min(norm_cam)
                    norm_cam = (norm_cam - np.min(norm_cam))/_range
                    norm_cam=norm_cam*255
                    norm_cam=cv2.resize(norm_cam,(width,highth))
                    output_map[int(x)-width//2:int(x)+width//2,int(y)-highth//2:int(y)+highth//2]=np.maximum(norm_cam,\
                                output_map[int(x)-width//2:int(x)+width//2,int(y)-highth//2:int(y)+highth//2])
                    prob_map[int(x)-width//2:int(x)+width//2,int(y)-highth//2:int(y)+highth//2]=np.maximum(pre[0],\
                                prob_map[int(x)-width//2:int(x)+width//2,int(y)-highth//2:int(y)+highth//2])
        
        
        
        output_map=output_map>200
        output_map=morphology.remove_small_objects(output_map,min_size=25,connectivity=1)
        output_map=output_map*255 
        label_output =measure.label(output_map)
        regions_output=measure.regionprops(label_output)
        img = cv2.imread(image_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        for region_output in regions_output:
            y_min=region_output.bbox[0]
            y_max=region_output.bbox[2]
            x_min=region_output.bbox[1]
            x_max=region_output.bbox[3]
            x=(x_min+x_max)//2
            y=(y_min+y_max)//2
            prob=prob_map[(y_min+y_max)//2,(x_min+x_max)//2]
            p1=(x-40,y-40)
            p2=(x+40,y+40)
            color_box=(0,0,255)
            line=(int(x),int(y),prob)
            with open(txt_output +'/'+basename+ '.txt', 'a') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')


            with open(label_file,encoding='utf-8')as fp:
                reader = csv.reader(fp)
                for point in reader:
                    if((int(point[0])-x)**2+(int(point[1])-y)**2<900):
                        color_box=(0,255,0)
                        break

            cv2.rectangle(img, p1, p2, color_box, 5, cv2.LINE_AA)
            cv2.putText(img,str(format(prob,".2f")),p1,0,2,color=color_box, thickness=3)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(savepath+'/'+basename+'.png', img.astype('uint8'))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=300, type=int)
    parser.add_argument("--thor", default=0.5, type=float)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--weights", default='checkpoint/best.pth', type=str)
    parser.add_argument("--input_path", default='path', type=str)
    parser.add_argument("--output_path", default='output_path', type=str)
    args = parser.parse_args()

    test_phase(args)







    
