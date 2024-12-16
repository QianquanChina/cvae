import torch
from torch.utils.data import DataLoader

from models import CVae
from library import train_model
from dataset import DATA_TRANSFORM, MnistDataset


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 数据处理
    train_dir = "/data/Vae/Dataset/MNIST_data/train"
    val_dir = "/data/Vae/Dataset//MNIST_data/test"
    save_dir = "/data/CVae/Weight"
    train_dataset = MnistDataset(image_dir=train_dir, transformer=DATA_TRANSFORM["train"])
    val_dataset = MnistDataset(image_dir=val_dir, transformer=DATA_TRANSFORM["val"])
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=4)

    # 模型加载
    model = CVae(in_channels=1, num_classes=10, embed_dim=10, latent_dim=20)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
    train_model(model, train_loader, val_loader, optimizer, 30, device, save_dir)

if __name__ == '__main__':
    main()
