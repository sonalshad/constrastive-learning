import torch
from tqdm import tqdm

def model_train(train_dataloader,
                val_dataloader,
                model,
                criterion,
                optimizer,
                epochs=10, 
                device = 'cpu'): 

    train_loss_log, val_loss_log = [], []

    for epoch in range(epochs): 
        train_loss = 0
        train_size = 0

        val_loss = 0
        val_size = 0
        
        # Training 
        for step, (x1, x2, y) in enumerate(train_dataloader): 
            
            model.train()
            
            x1 = x1.to(device)
            x2 = x2.to(device)

            train_size += y.shape[0]

            optimizer.zero_grad()
            z, labels = model(x1, x2)
            loss = criterion(z, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*y.shape[0]

        # Validation 
        for x1, x2, y in val_dataloader:
            
            model.eval()
            x1 = x1.to(device)
            x2 = x2.to(device)

            val_size += y.shape[0]

            z, labels = model(x1, x2)
            loss = criterion(z, labels)
            val_loss += loss.item()*y.shape[0]

        train_loss_log.append(train_loss/train_size)
        val_loss_log.append(val_loss/val_size)
        print(f"Epoch: {epoch+1}/{epochs} \t T Loss: {(train_loss/train_size):0.4f}\
          \t V Loss: {(val_loss/val_size):0.4f}")
    return train_loss_log, val_loss_log

def save_model(model, optimizer, name):
    out = os.path.join('/home/sshad/wcl-project/saved_models', name)

    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, out)