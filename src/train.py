"""
Train a SegNet model


Usage:
python train.py --data_root /home/SharedData/intern_sayan/PascalVOC2012/data/VOCdevkit/VOC2012/ \
                --train_path ImageSets/Segmentation/train.txt \
                --img_dir JPEGImages \
                --mask_dir SegmentationClass \
                --save_dir /home/SharedData/intern_sayan/PascalVOC2012/ \
                --checkpoint /home/SharedData/intern_sayan/PascalVOC2012/model_best.pth \
                --gpu 1
"""

from __future__ import print_function
import argparse
from dataset import PascalVOCDataset, NUM_CLASSES
from model import SegNet
import os
import time
import torch
from torch.utils.data import DataLoader
from dataloader import DataLoader as OwnDataLoader


# Constants
NUM_INPUT_CHANNELS = 3
NUM_OUTPUT_CHANNELS = 1

NUM_EPOCHS = 60

LEARNING_RATE = 1e-6
MOMENTUM = 0.9
BATCH_SIZE = 8


# Arguments
parser = argparse.ArgumentParser(description='Train a SegNet model')

# parser.add_argument('--data_root', required=True)
# parser.add_argument('--train_path', required=True)
# parser.add_argument('--img_dir', required=True)
# parser.add_argument('--mask_dir', required=True)
# parser.add_argument('--save_dir', required=True)
parser.add_argument('--checkpoint')
parser.add_argument('--gpu',default=1, type=int)
parser.add_argument('--cs', dest='cs', default='rgb', type=str, help='color space: rgb, lab')
parser.add_argument('--dir', dest='dir', default='./dataset/canopy_mask_dataset/group1/', type=str, help='dataset directory')
parser.add_argument('--s', dest='session', default=1, type=int, help='training session')

args = parser.parse_args()



def train():
    is_better = True
    prev_loss = float('inf')

    model.train()

    for epoch in range(NUM_EPOCHS):
        loss_f = 0
        t_start = time.time()

        for batch in train_dataloader:
            input_tensor = torch.autograd.Variable(batch['image'])
            target_tensor = torch.autograd.Variable(batch['mask'])

            if CUDA:
                input_tensor = input_tensor.cuda()
                target_tensor = target_tensor.cuda()

            predicted_tensor, softmaxed_tensor = model(input_tensor)
            optimizer.zero_grad()
            loss = criterion(predicted_tensor, target_tensor)
            loss.backward()
            optimizer.step()


            loss_f += loss.float()
            prediction_f = softmaxed_tensor.float()

        delta = time.time() - t_start
        is_better = loss_f < prev_loss

        if is_better:
            prev_loss = loss_f
            torch.save(model.state_dict(),  "model_best_s"+str(args.session)+".pth")

        print("Epoch #{}\tLoss: {:.8f}\t Time: {:2f}s".format(epoch+1, loss_f, delta))


if __name__ == "__main__":
    # data_root = args.data_root
    # train_path = os.path.join(data_root, args.train_path)
    # img_dir = os.path.join(data_root, args.img_dir)
    # mask_dir = os.path.join(data_root, args.mask_dir)

    CUDA = args.gpu is not None

    # train_dataset = PascalVOCDataset(list_file=train_path,
    #                                  img_dir=img_dir,
    #                                  mask_dir=mask_dir)

    # train_dataloader = DataLoader(train_dataset,
    #                               batch_size=BATCH_SIZE,
    #                               shuffle=True,
    #                               num_workers=4)
    data_dir=args.dir
    train_dataset = OwnDataLoader(root=data_dir,train=True,cs=args.cs)
    train_size = len(train_dataset)
    eval_dataset = OwnDataLoader(root=data_dir,train=False,cs=args.cs)
    eval_size = len(eval_dataset)
    print(train_size)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=4)
    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE,
                            shuffle=True, num_workers=4)
                            
    model = SegNet(input_channels=NUM_INPUT_CHANNELS,output_channels=NUM_OUTPUT_CHANNELS)
    criterion = torch.nn.KLDivLoss()
    if CUDA:
        model=model.cuda()
        # class_weights = 1.0/train_dataset.get_class_probability().cuda()
        criterion = criterion.cuda()


    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))


    optimizer = torch.optim.Adam(model.parameters(),
                                     lr=LEARNING_RATE)


    train()
