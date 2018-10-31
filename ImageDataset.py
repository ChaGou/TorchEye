import torch
import torch.utils.data
from PIL import Image
import math
import torchvision
from torchvision import transforms
def default_loader(path):
    return Image.open(path).convert('L')
transform1 = transforms.Compose([
    transforms.ToTensor(),
    ]
)
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, txt_path,transform = None, target_transform=None, loader=default_loader):
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            self.label_list = [i for i in lines]
            self.transform = transform
            self.target_transform = target_transform
            self.loader = loader
            self.img_path = img_path

    def __getitem__(self, index):
        label = self.label_list[index].split('\n')[0]
        label = torch.Tensor([float(label.split()[0]), float(label.split()[1])]).view(2,1)
        img = self.loader(self.img_path+str(index)+'.jpg')
        return transform1(img).view(1,400,400), label

    def __len__(self):
        return len(self.label_list)

# test = ImageLoader('images/', 'testfile.txt')
# l,im = test[2]
# import matplotlib.pyplot as plt
# plt.imshow(transform1(im).view(400,400))
# plt.show()
# print()