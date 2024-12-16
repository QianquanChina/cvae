import torch
from tqdm import tqdm

def train_model(model, train_loader, val_loader, optimizer, epochs, device, save_dir):
    model.train()
    model.to(device)
    for epoch in range(epochs):
        train_loss = 0
        for data in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images, labels)
            loss_dict = model.loss_function(*outputs, M_N=0.00025)
            loss = loss_dict["loss"]
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)


        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images, labels)
                loss_dict = model.loss_function(*outputs, M_N=0.00025)
                val_loss += loss_dict["loss"]

        avg_val_loss = val_loss / len(val_loader)
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{save_dir}/vae_epoch_dim20_{epoch+1}.pth")
        print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

