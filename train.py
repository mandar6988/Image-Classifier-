import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def main():
        input = take_args()
        train_network()
        data_dir = input.data_directory
        save_to = input.save_dir
        pretrained_model = input.arch
        learning_rate = input.lr_rate
        ep = input.epochs
        hidden_layers = input.hd_nodes
        output_size = input.output
        gpu = input.gpu
        
        
        
def pre_pro(data_dir):
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'

        data_transforms_train = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
        data_transforms_test=transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
        data_transforms_validation=transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


        train_data = datasets.ImageFolder(train_dir, transform=data_transforms_train)

        test_data = datasets.ImageFolder(test_dir, transform=data_transforms_test)

        validation_data=datasets.ImageFolder(valid_dir, transform=data_transforms_validation)
        print("ok")



        trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
        testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
        validloader=torch.utils.data.DataLoader(validation_data, batch_size=64)
        return trainloader,testloader,validloader

def train_network():
    input = take_args()
    data_dir = input.data_directory
    print(data_dir)
    trainloader,testloader,validloader=pre_pro(data_dir)
    pretrained_model = input.arch
    if pretrained_model=="densenet121":
         model = models.densenet121(pretrained=True)
         model
         input_features=1024        
         print("densenet")
    elif pretrained_model=="vgg19":
        model = models.vgg19(pretrained=True)
        model
        input_features=25088
        print("vgg19")
    else:
        print("wrong model selected")
       
    from collections import OrderedDict
    hd_nodes=input.hd_nodes
    
    input_nodes=input_features
    hd_nodes.insert(0,input_nodes)
    length=len(hd_nodes)
    hd_nodes.insert(length,input.output)
    print(hd_nodes)
    length=len(hd_nodes)
    print(length)
    modul_cntnr = OrderedDict()
    for i in range (length ):
        if i==(length-2):
            modul_cntnr['output'] = nn.Linear(hd_nodes[i ],input.output)
            modul_cntnr['softmax'] = nn.LogSoftmax(dim=1)
            break
        else:
            modul_cntnr['fc' + str(i + 1)] = nn.Linear(hd_nodes[i], hd_nodes[i + 1])
            modul_cntnr['relu'+str(i + 1)] = nn.ReLU()
            modul_cntnr['drpot'+str(i + 1)] = nn.Dropout(input.drop)
            print(i)
    classifier = nn.Sequential(modul_cntnr)
    print(classifier)
    criterion = nn.NLLLoss()   
    model.classifier = classifier
    optimizer = optim.Adam(model.classifier.parameters(), lr=input.lr_rate)   
    epochs = input.epochs
    steps = 0
    running_loss = 0
    print_every = 5
    model.to('cuda')
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to('cuda:0'), labels.to('cuda:0')
                        model.to('cuda:0')
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Valid loss: {valid_loss/len(validloader):.3f}.. "
                      f"Valid accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    model.class_to_idx = trainloader.dataset.class_to_idx
    checkpoint = {'input_size': input_features,
                  'output_size': input.output,
                  'hidden_nodes':input.hd_nodes,
                  'class_to_idx': model.class_to_idx,
                  'drop':input.drop,
                  'epochs':input.epochs,
                  'learning_rate':input.lr_rate,
                  'arch': input.arch,
                  'state_dict': model.state_dict()}

    torch.save(checkpoint,input.save_dir)            
 
def take_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("data_directory", type=str, help="it is path of training and testing data")
	parser.add_argument("--save_dir", type=str, default="checkpoint1.pth",
		                        help="trained model and network parameters saved in this directory. Please provide extension'.pth' after name of save directory")
	parser.add_argument("--arch", type=str, default="densenet121",
		                        help="pretrained models avaliable are vgg19, densenet121")
	parser.add_argument("--lr_rate", type=float, default=0.001,
	                        help="to select learning rate")
	parser.add_argument("--epochs", type=int, default=3,
		                        help="to select number of epochs to train model")
	parser.add_argument("--hd_nodes", type=list, default=[500, 250],
		                        help="contains no. of nodes with hidden layers")
	parser.add_argument("--gpu", type=bool, default=True,
		                        help="to select GPU(True) or CPU(False) ")
	parser.add_argument("--output", type=int, default=102,
	                        help="to select output size,")
	parser.add_argument("--drop", type=float, default=0.2,
	                        help="to select dropout probability,")
	return parser.parse_args()

    
    

if __name__ == "__main__":
	main()
