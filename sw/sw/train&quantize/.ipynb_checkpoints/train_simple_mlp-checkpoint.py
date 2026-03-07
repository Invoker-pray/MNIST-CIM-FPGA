import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(784,128,bias=True)
        self.relu=nn.ReLU()

    def forward(self,x):
        x=x.view(-1,784)
        x=self.fc1(x)
        x=self.relu(x)
        return x

def train():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,))
    ])

    train_dataset=datasets.MNIST('./data',train=True, download=True,transform=transform)
    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=64,shuffle=True)

    model=SimpleMLP().to(device)
    optimizer=optim.Adam(model.parameters(),lr=0.001)
    criterion=nn.MSELoss()

    for epoch in range(10):
        for batch_idx, (data,target) in enumerate(train_loader):
            data=data.to(device)
            optimizer.zero_grad()
            output=model(data)

            loss=output.pow(2).mean()
            loss.backward()
            optimizer.step()

            if batch_idx %100 ==0:
                print(f'Epoch {epoch},Batch {batch_idx}, Loss:{loss.item():.4f}')


    torch.save(model.state_dict(), 'simple_mlp.pth')
    return model

if __name__=='__main__':
    model=train()

