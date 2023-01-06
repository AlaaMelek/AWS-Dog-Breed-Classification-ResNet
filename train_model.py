#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import sys
import argparse
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

#TODO: Import dependencies for Debugging andd Profiling
try:
    import smdebug.pytorch as smd
except:
    pass

def test(model, test_loader, criterion, hook):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    if hook:
        hook.set_mode(smd.modes.EVAL)
    test_loss=0
    correct=0
    with torch.no_grad():
        for data, labels in test_loader:
            output = model(data) 
            loss = criterion(output, labels)
            test_loss+=loss
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

def valid(model, valid_loader, criterion, hook):
    model.eval()
    if hook:
        hook.set_mode(smd.modes.EVAL)
    valid_loss=0
    correct=0
    with torch.no_grad():
        for data, labels in valid_loader:
            output = model(data)
            loss = criterion(output, labels)
            valid_loss+=loss
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(labels.view_as(pred)).sum().item()

    valid_loss /= len(valid_loader.dataset)
    logger.info(
        "Valid set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            valid_loss, correct, len(valid_loader.dataset), 100.0 * correct / len(valid_loader.dataset)
        )
    )
        

def train(model, train_loader, criterion, optimizer, valid_loader, epochs, hook):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        if hook:
            hook.set_mode(smd.modes.TRAIN)
        for batch_idx, (data, labels) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
        valid(model, valid_loader, criterion, hook)
    return model
                
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model=models.resnet50(pretrained=True)
    # freeze the convolution part
    for param in model.parameters():
        param.requires_grad = False
    # find the number of inputs to the final layer of the network
    num_inputs = model.fc.in_features
    # modify the fully connected part for our problem
    model.fc = nn.Linear(num_inputs, 133)
    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    dataloaders = {
        split : torch.utils.data.DataLoader(data[split], batch_size, shuffle=True)
        for split in ['train', 'valid', 'test']
    }

    return dataloaders

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    # hook for depugging and profiling
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), args.lr)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    # datasets tranforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
    }
    
    # dataset loaders
    image_datasets = {
        split : datasets.ImageFolder(os.path.join(args.data_dir, split), data_transforms[split])
        for split in ['train', 'valid', 'test']
    }

    dataloaders = create_data_loaders(image_datasets , args.batch_size)
    train_loader = dataloaders['train']
    valid_loader = dataloaders['valid']
    test_loader = dataloaders['test']
    
    model=train(model, train_loader, loss_criterion, optimizer, valid_loader, args.epochs, hook)
        
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, loss_criterion, hook)
    
    '''
    TODO: Save the trained model
    '''
    path = os.path.join(args.model_dir, 'model.pth')
    torch.save(model, path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    # epoch
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        metavar="E",
        help="number of epochs to train (default: 2)",
    )
    # batch_size
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 32)",
    )
    # lr
    parser.add_argument(
        "--lr", type=float, default=0.1, metavar="LR", help="learning rate (default: 0.1)"
    )

    # Container environment
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_DATA"])
    parser.add_argument('--output-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args=parser.parse_args()
    
    main(args)
