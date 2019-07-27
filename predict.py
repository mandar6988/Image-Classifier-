import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import json


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
def main():
    input = take_args()
    print(input.load_checkpoint)
    checkpoint = torch.load(input.load_checkpoint)
    print(checkpoint['input_size'])
    model=checkpoint['arch']
    loaded_model, class_to_idx=load_checkpoint(input.load_checkpoint)
    idx_to_class = dict((val,key) for key,val in class_to_idx.items())
    topk=input.k_class
    print(topk)
    img = input.take_image
    img1=Image.open(img)
    p=process_image(img1)
   
    five_probs,five_class=predict(img,loaded_model,idx_to_class,topk)
    flower_name = [cat_to_name[i] for i in five_class]
    print("First '{:d}' Flower Types are".format(topk))
    print(flower_name)
    print("first '{:d}' Probabilities are".format(topk))
    print(five_probs)
    print("first '{:d}' Classes are".format(topk))
    print(five_class)
    
    
def load_checkpoint(filepath):    
    input = take_args()
    checkpoint = torch.load(filepath)
    pretrained_model = checkpoint['arch']
    if pretrained_model=="densenet121":
         model = models.densenet121(pretrained=True)
         input_features=1024        
         print("densenet121")
    elif pretrained_model=="vgg19":
        model = models.vgg19(pretrained=True)
        input_features=25088
        print("vgg19")
    else:
        print("wrong model selected")
    hd_nodes=checkpoint['hidden_nodes']  
    drop=checkpoint['drop']
    print(hd_nodes)
    model.class_to_idx = checkpoint['class_to_idx']
    
    from collections import OrderedDict
    length=len(hd_nodes)
    length=len(hd_nodes)
    modul_cntnr = OrderedDict()
    for i in range (length ):
        if i==(length-2):
            modul_cntnr['output'] = nn.Linear(hd_nodes[i ],102)
            modul_cntnr['softmax'] = nn.LogSoftmax(dim=1)
            break
        else:
            modul_cntnr['fc' + str(i + 1)] = nn.Linear(hd_nodes[i], hd_nodes[i + 1])
            modul_cntnr['relu'+str(i + 1)] = nn.ReLU()
            modul_cntnr['drpot'+str(i + 1)] = nn.Dropout(drop)
            print(i)
    classifier = nn.Sequential(modul_cntnr)
    
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    
    return model, checkpoint['class_to_idx']
    
def process_image(image):
    prepro_img=transforms.Compose([transforms.Resize(255),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    img1=np.array(prepro_img(image))
    return img1    
    
def predict(image_path, model,idx_to_class,topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    img=Image.open(image_path)
    t_image=process_image(img)
    t_image=torch.FloatTensor([t_image])
    model.eval()
    output = model.forward(Variable(t_image))
    ps = torch.exp(output)
    ps=ps.detach().numpy()
    ps=ps.reshape(102,)
    chk=np.argsort(ps)
    first_five_index=chk[len(ps)-(topk):(len(ps))][::-1]
     
    first_five_class = [idx_to_class[i] for i in first_five_index]
    top_probability = ps[first_five_index]
   
    return top_probability, first_five_class
   
    # TODO: Implement the code to predict the class from an image file
 
    


def take_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--load_checkpoint",type=str,default="checkpoint5.pth",help="to load trained model, Default =checkpoint5 (vgg19) avalaible=checkpoint5.pth(vgg19), checkpoint1.pth(densenet121) please provide'.pth'extension for other case")
	parser.add_argument("--take_image", type=str, default='flowers/test/74/image_01209.jpg',
	                        help="enter a flower string to classify its class.   ....e.g='flowers/test/74/image_01209.jpg'")
                               
	parser.add_argument("--k_class", type=int, default=8,
	                        help="enter number to print that numbers's probabilities")
	parser.add_argument("--gpu", type=bool, default=True,
	                  help="to select GPU(True) or CPU(False) ")
	
	
	
	return parser.parse_args()

if __name__ == "__main__":
	main()