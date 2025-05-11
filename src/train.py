import torch
import torch.nn as nn
import torch.optim as optim

def train_model(model, train_loader, val_loader, num_epochs=10, device='cpu'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    history = {'train_loss': [], 'val_loss': []}
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device, dtype=torch.float)
            masks = masks.to(device, dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_loss)

        # 検証
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device, dtype=torch.float)
                masks = masks.to(device, dtype=torch.long)
                outputs = model(images)['out']
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
        val_epoch_loss = val_loss / len(val_loader.dataset)
        history['val_loss'].append(val_epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}")
    return model, history 