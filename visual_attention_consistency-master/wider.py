import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import glob



class CustomDataset(Dataset):
    def __init__(self, dataset_path, scale, mode='train'):
        super().__init__()
        self.mode = mode
        self.size = scale

        self.img_path = os.path.join(dataset_path, 'data', 'img')
        self.mask_path = os.path.join(dataset_path, 'data', 'mask')
        self.image_lists = self.read_list(self.img_path)
        self.resize_img = transforms.Resize(scale, Image.BILINEAR)

        self.to_tensor = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize(mean = [0.485, 0.456, 0.406],
				std = [0.229, 0.224, 0.225]),
			])


        self.classification_info = self.load_classification_info(self.mask_path)

    def __getitem__(self, index):
        img = Image.open(self.image_lists[index]).convert('RGB')
        img = self.resize_img(img)
        img = np.array(img)
        img = self.to_tensor(img.copy()).float()

        filename = os.path.basename(self.image_lists[index])
        classification_info = self.classification_info[filename]
        classification_info = [int(value) for value in classification_info]
        classification_info = torch.tensor(classification_info, dtype=torch.float32)

        return img, classification_info

    def __len__(self):
        return len(self.image_lists)

    def read_list(self, image_path):
        image_path = os.path.join(image_path, self.mode)
        img_list = glob.glob(os.path.join(image_path, '*.png'))
        return img_list

    def load_classification_info(self, mask_path):
        mask_path = os.path.join(mask_path, self.mode)
        info_path = os.path.join(mask_path, f'multi_classification_{self.mode}.txt')

        classification_info = {}
        with open(info_path, 'r') as txt_file:
            for line in txt_file:
                filename, *info = line.strip().split(',')
                classification_info[filename] = list(map(int, info))

        return classification_info


def get_subsets(dataset_path, scale):
	trainset = CustomDataset(dataset_path, scale, 'train')
	valset = CustomDataset(dataset_path, scale, 'val')
	testset = CustomDataset(dataset_path, scale, 'test')
	return trainset, valset, testset

if __name__ == '__main__':

	dataset_path = r'E:\MY_DATASET\CLASSIFICATION(ENHANCE)\ILMD'
	trainset, valset, testset = get_subsets(dataset_path, (256, 512))

	train_loader = torch.utils.data.DataLoader(
		trainset,
		batch_size=1,
		shuffle=False,
		num_workers=8,
		)
	val_loader = torch.utils.data.DataLoader(
		valset,
		batch_size=1,
		shuffle=False,
		num_workers=2,
	)
	test_loader = torch.utils.data.DataLoader(
		testset,
		batch_size=1,
		shuffle=False,
		num_workers=2,
		)

	# 查看训练集中的前4个样本
	for i, (images, labels) in enumerate(train_loader):
		if i < 4:
			print("Trainset infomation Testing")
			print(f"Training Sample {i + 1}:")
			image_name = os.path.basename(trainset.image_lists[i])
			print(f"image: {image_name}")
			print("Image shape:", images.shape)
			print("Label:", labels[0].numpy())
			print("-----------------------------")
		else:
			break

	# 查看验证集集中的前4个样本
	for i, (images, labels) in enumerate(val_loader):
		if i < 4:
			print("Valset infomation Testing")
			print(f"Testing Sample {i + 1}:")
			image_name = os.path.basename(trainset.image_lists[i])
			print(f"image: {image_name}")
			print("Image shape:", images.shape)
			print("Label:", labels[0].numpy())
			print("-----------------------------")
		else:
			break

	# 查看测试集中的前4个样本
	for i, (images, labels) in enumerate(test_loader):
		if i < 4:
			print("Testset infomation Testing")
			print(f"Testing Sample {i + 1}:")
			image_name = os.path.basename(trainset.image_lists[i])
			print(f"image: {image_name}")
			print("Image shape:", images.shape)
			print("Label:", labels[0].numpy())
			print("-----------------------------")
		else:
			break