import torch
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, idx):
        image = torch.ones(3, 4, 4) * idx
        cond = {"y": idx}
        return image, cond


dataset = MyDataset()
loader = DataLoader(dataset, batch_size=3)

for images, cond in loader:
    print("images type:", type(images))
    print("images shape:", images.shape)
    print("cond type:", type(cond))
    print("cond:", cond)
    print("cond['y'] type:", type(cond["y"]))
    print("cond['y'] shape:", cond["y"].shape)
    break