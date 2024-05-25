import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearHead(nn.Module): 
    
    # resnet class outputs 2048 dim embedding
    # CIFAR has 10 classes
    def __init__(self, net, dim_in=2048, dim_out=10): 
        super().__init__()
        
        self.net = net
        #Freeze layers 
        for param in self.net.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(dim_in, dim_out)

    def forward(self, x): 
        with torch.no_grad(): 
            feat = self.net(x)
        return self.fc(feat)

def evaluate_model(train_loader, test_loader, model, optimizer, epochs=10, device='cpu'): 
    
    model = model.to(device)
    for epoch in range(epochs): 
        
        train_loss = 0
        train_size = 0

        for image, y in test_loader: 
            
            image = image.to(device)
            y = y.to(device).long()


            out = model(image)
            loss = F.cross_entropy(out, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()*y.size(0)
            train_size += y.size(0)

        with torch.no_grad(): 
            
            total = 0
            correct = 0
            
            for image, y in test_loader: 
                image = image.to(device)
                y = y.to(device)

                out = model(image)
                preds = torch.argmax(out, dim=1)
                total += y.size(0)
                correct += (y==preds).sum().item()

        print(f"Epoch {epoch+1}/{epochs} Train Loss = {train_loss/train_size} \
        Test Accuracy = {correct/total}")

