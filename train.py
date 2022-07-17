import pandas as pd
import numpy as np
import cv2
import os
from os.path import exists
import re

from PIL import Image

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch

import torchvision
from torchvision import transforms
from PIL import Image

from sklearn.model_selection import train_test_split
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SequentialSampler

#from matplotlib import pyplot as plt


#plt.ion()   # interactive mode
#%matplotlib inline


########


def collate_fn(batch):
    return tuple(zip(*batch))


def get_train_transform():
    return A.Compose([
        A.Flip(0.5),
        ToTensorV2(p=1.0),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


class MOTDataset(Dataset):

    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe['frame_no']  # .unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __getitem__(self, index: int):
        #         print(index)
        #         print( self.image_ids)
        image_id = self.image_ids[index]
        # image_ids=image_ids.apply(str)
        image_idst = str(image_id)
        c = len(image_idst)
        a = 6 - c
        image_idst = '0' * a + image_idst
        records = self.df[self.df['frame_no'] == int(image_id)]
        # print("records")
        # print(records)
        # print(image_id)
        path = self.image_dir + '/' + image_idst + '.jpg'
        # transform = torchvision.transforms.ToTensor()

        image = cv2.imread(path, cv2.IMREAD_COLOR)

        #plt.imshow(image)
        image = torch.Tensor(image.reshape(1, image.shape[2], image.shape[0], image.shape[1]))
        # image=torch.FloatTensor(image)

        boxes = records[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values
        # print(boxes.shape)
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # boxes=boxes.resize(1,4)

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area, dtype=torch.float32)

        labels = torch.tensor(self.df['Class'].values, dtype=torch.int64)
        # images = list(img for img in image)
        # print(type(labels))
        targets = []

        for i in range(len(boxes)):
            d = {}
            d['boxes'] = boxes[i].reshape(1, 4)
            l = labels[i]
            d['labels'] = l.reshape(1)
            targets.append(d)
            # print()
        return image, targets, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]


c = 0
train_dataset = {}
valid_dataset = {}

# DIR_INPUT = 'MOT20Det/train/MOT20-0'+str(1)
for i in [1, 2, 3, 5]:
    DIR_INPUT = 'MOT20Det/train/MOT20-0' + str(i)
    DIR_TRAIN = f'{DIR_INPUT}/img1'
    # DIR_TEST = f'{DIR_INPUT}/test'
    train_df = pd.DataFrame(pd.read_csv(DIR_INPUT + '/gt/gt.txt', delimiter=',', header=None))  # (f'{DIR_INPUT}/img1')
    train_df.rename(
        columns={0: "frame_no", 1: "object_id", 2: "bb_left", 3: "bb_top", 4: "bb_width", 5: "bb_height", 6: "score",
                 7: "Class", 8: "Visibility"}, inplace=True)
    # train_df

    # image_ids=pd.DataFrame(image_ids)
    image_ids = train_df['frame_no']  # .unique()
    train_ids, valid_ids = train_test_split(image_ids, test_size=0.2)

    valid_df = train_df[train_df['frame_no'].isin(valid_ids)]
    train_df = train_df[train_df['frame_no'].isin(train_ids)]

    train_dataset[i] = MOTDataset(train_df, DIR_TRAIN, get_train_transform())
    valid_dataset[i] = MOTDataset(valid_df, DIR_TRAIN, get_valid_transform())
    indices = torch.randperm(len(train_dataset)).tolist()
    c = c + 1

train_data_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

valid_data_loader = DataLoader(
    valid_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn
)

print('Data Loaded Properly')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.000001, momentum=0.7, weight_decay=0.000005)
# lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
lr_scheduler = None

num_epochs = 3
loss_hist = Averager()
itr = 1
for epoch in range(num_epochs):

    loss_hist.reset()
    for i in train_data_loader.dataset:
        for images, targets, image_ids in train_data_loader.dataset[i]:

            # images = list(image for image in images)
            # targets = targets[{k: v for k, v in t} for t in targets]
            # print(images.shape)
            model.train()
            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            loss_hist.send(loss_value)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            if itr % 50 == 0:
                print(f"Iteration #{itr} loss: {loss_value}")
                with torch.no_grad():
                    loss_dict_val = model(images, targets)

                    losses_val = sum(loss for loss in loss_dict_val.values())
                    loss_Val_value = losses_val.item()
                    print("val", loss_Val_value)
            itr += 1


    # update the learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()

    print(f"Epoch #{epoch} loss: {loss_hist.value}")


## Save model
save_prefix = os.path.join('', "Faster_RCNN")
save_path = '{}_epoch_{}.pt'.format(save_prefix, epoch)
print("save all model to {}".format(save_path))
output = open(save_path, mode="wb")
torch.save(model.state_dict(), save_path)
model.eval()

## Validate


