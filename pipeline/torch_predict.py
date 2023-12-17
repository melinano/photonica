from skimage import io
import skimage
import matplotlib.pyplot as plt
import os
import torch
from skimage.util import img_as_ubyte
from torchvision import transforms
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryAccuracy
from torch_metrics import DiceMetric, DiceBCELoss
import numpy as np
from PIL import Image

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model


def pred_one_file(image_path, model_path):
    trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    model = load_checkpoint(model_path)
    model = model.to('cpu')
    try:
        model = model.module
    except AttributeError:
        model = model

    image = io.imread(image_path)
    image = skimage.img_as_ubyte(image)  # if float img type
    image = trans(image)
    image = torch.reshape(image, (1, 3, 512, 512))
    image = image.to('cpu')
    pred = model(image)
    pred = pred.cpu().numpy()[0][0]
    pred = (pred > 0.5) * 1
    plt.imshow(pred, cmap='gray')


def predict(images, model_path):
    trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    model = load_checkpoint(model_path)
    model = model.to('cpu')
    pred_images = list()
    try:
        model = model.module
    except AttributeError:
        model = model

    for i, image in enumerate(images):
        print(f'{i}/{len(images)}')
        image = trans(image)
        image = torch.reshape(image, (1, 3, 512, 512))
        image = image.to('cpu')
        pred = model(image)
        pred = (pred > 0.5) * 1

        pred = pred.cpu().numpy()[0][0]


        pred_img =Image.fromarray((pred * 255).astype(np.uint8))
        pred_images.append(pred_img)
        #pred_img.save(out_path + '/' + image_name)
        continue
    return pred_images

def predict_from_dir(image_path, model_path, out_path):
    trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    model = load_checkpoint(model_path)
    model = model.to('cpu')
    try:
        model = model.module
    except AttributeError:
        model = model

    image_list = os.listdir(image_path)  # [1:7]
    image_list.sort()
    print(image_list)

    for i, image_name in enumerate(image_list):
        print(f'{i}/{len(image_list)}')
        image_name = '.'.join(image_name.split(sep='.')[:-1]) + '.png'
        image = io.imread(os.path.join(image_path, image_list[i]))
        image = trans(image)
        image = torch.reshape(image, (1, 3, 512, 512))
        image = image.to('cpu')
        pred = model(image)
        pred = (pred > 0.5) * 1

        pred = pred.cpu().numpy()[0][0]


        pred_img =Image.fromarray((pred * 255).astype(np.uint8))
        pred_img.save(out_path + '/' + image_name)
        #io.imsave(out_path + '/' + image_name, pred, check_contrast=False)
        continue