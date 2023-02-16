import argparse
import cv2
import numpy as np
import os
import torch
from model import SegNet
import sys

NUM_INPUT_CHANNELS = 3
NUM_OUTPUT_CHANNELS = 1

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='canopy segmentation on video, creating another video')
    parser.add_argument('--cuda', dest='cuda', default=True, action='store_true', help='whether use CUDA')
    parser.add_argument('--input', dest='input', default='./dataset/input_images/flights/DJI_0607.mp4', type=str, help='path to a single input image for evaluation')
    parser.add_argument('--pred_folder', dest='pred_folder', default='./dataset/predicted_images/', type=str, help='where to save the predicted images.')
    parser.add_argument('--model_path', dest='model_path', default='saved_models/saved_model_1_9.pth', type=str, help='path to the model to use')
    parser.add_argument('--model_size', dest='model_size', default='large', type=str, help='size of the model: small, medium, large')
    parser.add_argument('--one_vid', dest='one_vid', default=True, type=bool, help='if you are processing multiple videos from a folder, do you want to create separate or only one video?')
    parser.add_argument('--frames', dest='frames', default=1, type=int, help='process every Xth frame from the video')
    parser.add_argument('--height', dest='height', default=480, type=int, help='height of the output video')
    parser.add_argument('--width', dest='width', default=640, type=int, help='width of the output video')
    parser.add_argument('--dim', dest='dim', default=False, type=bool, help='dim the pixels that are not segmented, or leave them black?')
    parser.add_argument('--cs', dest='cs', default='rgb', type=str, help='color space: rgb, lab')
    args = parser.parse_args()
    return args

def progress(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s --- %s/%s %s\r' % (bar, percents, '%', str(total), str(count), suffix))
    sys.stdout.flush()

if __name__ == '__main__':

    args = parse_args()
    isExist = os.path.exists(args.pred_folder)
    if not isExist:
        os.makedirs(args.pred_folder)
        print("The new directory for saving images while training is created!")
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: CUDA device is available. You might want to run the program with --cuda=True")
    if args.model_size not in ['small','medium','large']:
        print("WARNING. Model size of <%s> is not a valid unit. Accepted units are: small, medium, large. Defaulting to medium."%(args.model_size))
        args.model_size = 'medium'
    # network initialization
    print('Initializing model...')
    model = SegNet(input_channels=NUM_INPUT_CHANNELS, output_channels=NUM_OUTPUT_CHANNELS)
    if args.cuda:
        model = model.cuda()
    print("Model initialization done.")  
    
    load_name = os.path.join(args.model_path)
    print("loading checkpoint %s" % (load_name))
    model.load_state_dict(torch.load(args.model_path))
    # if 'pooling_mode' in checkpoint.keys():
    #     POOLING_MODE = checkpoint['pooling_mode']
    print("loaded checkpoint %s" % (load_name))
    # del checkpoint
    torch.cuda.empty_cache()
    model.eval()
    print('evaluating...')
    img_array = []
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    myFrameNumber = args.frames
    if args.cuda:
        tensorone = torch.Tensor([1.]).cuda()
        tensorzero = torch.Tensor([0.]).cuda()
    else:
        tensorone = torch.Tensor([1.])
        tensorzero = torch.Tensor([0.])
    with torch.no_grad():
        if args.input.endswith('.mp4'):
            if not os.path.exists(args.input):
                print("The file: "+args.input+" does not exists.")
                exit()
            dirname, basename = os.path.split(args.input)
            save_path=args.pred_folder+basename[:-4]
            print("processing: "+args.input)
            cap = cv2.VideoCapture(args.input)
            totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            print("total frames in video: "+str(totalFrames))
            fps = cap.get(cv2.CAP_PROP_FPS)
            video = cv2.VideoWriter(save_path+"_segnet.mp4", fourcc, fps, (args.width,args.height))
            currentFrame = 0
            while currentFrame<totalFrames-1:
                progress(currentFrame,totalFrames,"frames")
                cap.set(cv2.CAP_PROP_POS_FRAMES,currentFrame)
                ret, img = cap.read()
                imgmasked = img.copy()
                if args.cs=="lab":
                    try:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                    except:
                        print("Something went wrong processing bgr2lab conversion.")
                        break
                try:
                    img = cv2.resize(img,(args.width,args.height))
                    imgmasked = cv2.resize(imgmasked,(args.width,args.height))
                except:
                    print("Something went wrong processing resize.")
                    break
                img = np.moveaxis(img,-1,0)
                imgmasked = np.moveaxis(imgmasked,-1,0)
                img = torch.from_numpy(img).float().unsqueeze(0)
                imgmasked = torch.from_numpy(imgmasked).float().unsqueeze(0)
                if args.cs=="rgb":
                    img/=255.
                if args.cuda:
                    img = img.cuda()
                    imgmasked = imgmasked.cuda()
                maskpred, softmaxed_tensor = model(img)
                threshold = maskpred.mean()
                # imgmasked = img.clone()
                maskpred3=maskpred.repeat(1,3,1,1)
                if args.dim:
                    imgmasked[maskpred3<threshold]/=3
                else:
                    imgmasked[maskpred3<threshold]=tensorzero 
                outimage = imgmasked[0].cpu().detach().numpy()
                outimage = np.moveaxis(outimage,0,-1)
                # try:
                #         outimage = cv2.cvtColor(outimage, cv2.COLOR_LAB2RGB)*255
                # except:
                #     print("Something went wrong processing lab2bgr conversion.")
                #     break
                video.write(outimage.astype(np.uint8))
                currentFrame = currentFrame + myFrameNumber
            cap.release()
            video.release()
            print("done")
        else:
            if os.path.isfile(args.input):
                print("The specified file: "+args.input+" is not an mp4 video, nor a folder containing mp4 videos. If you want to evaluate images, use eval.py. If your videos are in different format than mp4, convert them or rewrite the code.")
                exit()
            if not os.path.exists(args.input):
                print("The folder: "+args.input+" does not exists.")
                exit()
            dlist=os.listdir(args.input)
            dlist.sort()
            fps = 0
            video = None
            counter = 0
            for filename in dlist:
                if filename.endswith(".mp4"):
                    counter = counter+1
                    print("processing: "+args.input+filename)
                    cap = cv2.VideoCapture(args.input+filename)
                    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    print("total frames in video: "+str(totalFrames))
                    if args.one_vid:
                        save_path=args.pred_folder+"full_segmented_video.mp4"
                    else:
                        save_path=args.pred_folder+filename[:-4]+"_segnet.mp4"
                    if video == None or not args.one_vid:
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        video = cv2.VideoWriter(save_path, fourcc, int(fps), (args.width,args.height))
                    currentFrame = 0
                    while currentFrame<totalFrames-1:
                        progress(currentFrame,totalFrames,"frames")
                        cap.set(cv2.CAP_PROP_POS_FRAMES,currentFrame)
                        ret, img = cap.read()
                        imgmasked = img.copy()
                        if args.cs=="lab":
                            try:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                            except:
                                print("Something went wrong processing bgr2lab conversion.")
                                break
                        try:
                            img = cv2.resize(img,(args.width,args.height))
                            imgmasked = cv2.resize(imgmasked,(args.width,args.height))
                        except:
                            print("Something went wrong processing resize.")
                            break
                        img = np.moveaxis(img,-1,0)
                        img = torch.from_numpy(img).float().unsqueeze(0)
                        imgmasked = np.moveaxis(imgmasked,-1,0)
                        imgmasked = torch.from_numpy(imgmasked).float().unsqueeze(0)
                        if args.cs=="rgb":
                            img/=255.
                        if args.cuda:
                            img = img.cuda()
                            imgmasked = imgmasked.cuda()
                        maskpred, softmaxed_tensor = model(img)
                        threshold = maskpred.mean()
                        maskpred3=maskpred.repeat(1,3,1,1)
                        if args.dim:
                            imgmasked[maskpred3<threshold]/=3
                        else:
                            imgmasked[maskpred3<threshold]=tensorzero
                        outimage = imgmasked[0].cpu().detach().numpy()
                        outimage = np.moveaxis(outimage,0,-1)
                        video.write(outimage.astype(np.uint8))
                        currentFrame = currentFrame + myFrameNumber
                    cap.release()
                    if not args.one_vid:
                        video.release()
                    print("done")
            if args.one_vid and video != None:
                video.release()
            if counter<1:
                print("The specified folder: "+args.input+" does not contain videos.")
