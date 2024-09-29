#!/usr/bin/env python
# coding: utf-8

# # Palmprint Recognition Network (PRN)
#
# In this notebook, one can test the functionality of the recognition network.
# Therefore install the environment.yaml file and you are ready to go!
#
# The whole Network is tested on the Tongji - Full palmprint dataset


import random
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable
from torchvision import transforms, models
from torch.utils.data import Dataset
import torchvision
from torchvision import datasets
import itertools
import matplotlib.pyplot as plt
import time
from PIL import Image, ImageDraw, ImageFont
import copy

torch.set_default_dtype(torch.float64)
import numpy as np
import cv2
from utils import getTongjiLabels
from models import ROILAnet
from models import TPSGridGen


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


current_script_dir = os.path.dirname(__file__)
checkpoints_dir = os.path.join(os.path.dirname(current_script_dir), "checkpoints")
ROIModelPath = os.path.join(checkpoints_dir, "ROI_extractor_augmented_TJ-NTU.pt")
CNNModelPath = os.path.join(checkpoints_dir, "resnet18_tongji_unfreezed_extractor.pt")


# for tongji we have 600 different palm of 300 different persons
class_names = getTongjiLabels()


# ## Defnition of Loaders


def loadROIModel(weightPath: str = None):
    """
    @weightPath: path to the ROILAnet() weights
    loads localization network with pretrained weights
    """
    model = ROILAnet()
    model.load_state_dict(torch.load(weightPath, map_location=torch.device(device)))
    model = model.to(device)
    model.eval()
    model.requires_grads = False
    return model


def getThinPlateSpline(
    target_width: int = 112, target_height: int = 112
) -> torch.Tensor:
    """
    @target_width: desired I_ROI output width
    @target_height: desired I_ROI output height
    greates instance of TPS grid generator
    """
    # creat control points
    target_control_points = torch.Tensor(
        list(
            itertools.product(
                torch.arange(-1.0, 1.00001, 1.0),
                torch.arange(-1.0, 1.00001, 1.0),
            )
        )
    )
    gridgen = TPSGridGen(
        target_height=target_height,
        target_width=target_width,
        target_control_points=target_control_points,
    )
    gridgen = gridgen.to(device)
    return gridgen


def getOriginalAndResizedInput(
    path: str = None,
) -> (np.ndarray, torch.Tensor, torch.Tensor):
    """
    @path: image which should be loaded from database
    This function load the image of variable size from a directory given in path.
    After doing the resizing to 56x56 pixels, the original and resized image will be returned
    as (PILMain, source_image, resizedImage) triplet
    """
    if path is None:
        return (None, None)

    # define transformer for resized input of feature extraction CNN
    resizeTranformer = transforms.Compose(
        [
            transforms.Resize((56, 56)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    PILMain = Image.open(path).convert(mode="RGB")  # load image in PIL format
    sourceImage = np.array(PILMain).astype("float64")  # convert from PIL to float64
    sourceImage = transforms.ToTensor()(sourceImage).unsqueeze_(
        0
    )  # add first dimension, which is batch dim
    sourceImage = sourceImage.to(device)  # load to available device

    resizedImage = resizeTranformer(PILMain)
    resizedImage = resizedImage.view(
        -1, resizedImage.size(0), resizedImage.size(1), resizedImage.size(2)
    )
    resizedImage = resizedImage.to(device)  # load to available device
    return (PILMain, sourceImage, resizedImage)


def getThetaHat(resizedImage: torch.Tensor = None, model=None) -> torch.Tensor:
    """
    @resizedImage: cropped image
    @model: ROI Localisation network, which outputs a theta vector
    resizedImage: image which should is loaded from database via getOriginalAndResizedInput function
    Here the theta vector is calculated using the pretrained localisation network. The vector has a size of
    [9, 2] -> which stand for 9 pairs of x and y values
    """
    if resizedImage is None:
        return None

    with torch.no_grad():  # deactivate gradients because we try to predict the ROI
        theta_hat = model.forward(resizedImage)
    theta_hat = theta_hat.view(
        -1, 2, 9
    )  # split into x and y vector -> theta_hat is originally a vector like [xxxxxxxxxyyyyyyyyy]
    theta_hat = torch.stack((theta_hat[:, 0], theta_hat[:, 1]), -1)
    return theta_hat


def sampleGrid(
    theta_hat: torch.Tensor = None,
    sourceImage: torch.Tensor = None,
    target_width: int = 112,
    target_height: int = 112,
) -> torch.Tensor:
    """
    @theta_hat: theta vector of normlized x,y coordinate pairs
    @sourceImage: the original image without any crops or resizsing
    @target_width: output IROI target width
    @target_height: output IROI target height
    Samples grid from a given theta vector, source image and grid generator
    """
    gridgen = getThinPlateSpline(target_width, target_height)
    # generate grid from calculated theta_hat vector
    source_coordinate = gridgen(theta_hat)
    # create target grid - with target height and target width
    grid = source_coordinate.view(-1, target_height, target_width, 2).to(device)
    # sample ROI from input image and created T(theta_hat)
    target_image = F.grid_sample(sourceImage, grid, align_corners=False)
    return target_image


# In[10]:


def printExtraction(target_image: torch.Tensor = None, source_image=None):
    """
    @source_image: prints the source_image which is in the PIL format
    @target_image: print the target_image which is a tensor (ROI)
    """
    # prepare to show -> get back from gpu if needed
    target_image = (
        target_image.cpu().data.numpy().squeeze().swapaxes(0, 1).swapaxes(1, 2)
    )
    target_image = Image.fromarray(target_image.astype("uint8"))
    plt.imshow(source_image)
    plt.show()  # show original image
    plt.imshow(target_image)
    plt.show()  # show ROI


def loadCNNModel(weightPath: str = None):
    """
    @weightPath: path to the ROILAnet() weights
    loads localization network with pretrained weights
    """
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load(weightPath, map_location=torch.device(device)))
    model.to(device)
    return model


def getIROI(model, input):
    resizedImage = F.interpolate(input, (56, 56))
    theta_hat = getThetaHat(
        resizedImage=resizedImage, model=model
    )  # create theta hat with normlized ROI coordinates
    IROI = sampleGrid(
        theta_hat=theta_hat, sourceImage=input, target_width=224, target_height=224
    )  # get ROI from source image
    IROI.to(device)
    return IROI


def markImage(image, theta):
    nimg = np.array(image)
    ocvim = cv2.cvtColor(nimg, cv2.COLOR_RGB2BGR)
    for idx, coord in enumerate(theta):
        currX = coord[0]
        currY = coord[1]
        x = int((ocvim.shape[1] - 1) / (1 + 1) * (currX - 1) + ocvim.shape[1])
        y = int((ocvim.shape[0] - 1) / (1 + 1) * (currY - 1) + ocvim.shape[0])
        ocvim = cv2.circle(ocvim, (x, y), 6, (200, 0, 0), 2)
        ocvim = cv2.putText(
            ocvim,  # numpy array on which text is written
            str(idx),  # text
            (x, y),  # position at which writing has to start
            cv2.FONT_HERSHEY_SIMPLEX,  # font family
            1,  # font size
            (209, 80, 0, 255),  # font color
            3,
        )
    ocvim = ocvim[..., ::-1]
    return ocvim


def getOriginalAndResizedInput(PILMain) -> (np.ndarray, torch.Tensor, torch.Tensor):
    """
    @path: image which should be loaded from database
    This function load the image of variable size from a directory given in path.
    After doing the resizing to 56x56 pixels, the original and resized image will be returned
    as (PILMain, source_image, resizedImage) triplet
    """
    if PILMain is None:
        return (None, None)

    # define transformer for resized input of feature extraction CNN
    resizeTranformer = transforms.Compose(
        [
            transforms.Resize((56, 56)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # PILMain = PILMain.convert(mode = 'RGB') # load image in PIL format
    sourceImage = np.array(PILMain).astype("float64")  # convert from PIL to float64
    sourceImage = transforms.ToTensor()(sourceImage).unsqueeze_(
        0
    )  # add first dimension, which is batch dim
    sourceImage = sourceImage.to(device)  # load to available device

    resizedImage = resizeTranformer(PILMain)
    resizedImage = resizedImage.view(
        -1, resizedImage.size(0), resizedImage.size(1), resizedImage.size(2)
    )
    resizedImage = resizedImage.to(device)  # load to available device
    return (PILMain, sourceImage, resizedImage)


# load localisation netowork
localisationNetwork = loadROIModel(ROIModelPath)  # load localisation network

# recognition Network setup pretrained
recognitionNetwork = loadCNNModel(CNNModelPath)


# # Prediction Area
def roi_data_agumentation(path: str):
    """
    Process a single image and return 6 different versions of it with advanced preprocessing.

    @path: path to the input image
    @returns: List of 6 preprocessed image tensors
    """
    recognitionNetwork.eval()
    input_PIL = Image.open(path).convert("RGB")
    grayscale = input_PIL.convert("L")
    grayscale_array = np.array(grayscale)
    rgb_array = np.stack((grayscale_array,) * 3, axis=-1)
    input_PIL = Image.fromarray(rgb_array)
    input_PIL.show()
    # Common preprocessing steps
    PILMain, sourceImage, resizedImage = getOriginalAndResizedInput(input_PIL)
    sourceImage = torch.stack([sourceImage.squeeze()])
    resizedImage = torch.stack([resizedImage.squeeze()])

    # Get normalized coordinates
    theta_hat = getThetaHat(resizedImage, localisationNetwork)

    # Get all ROIs
    IROI = sampleGrid(
        theta_hat=theta_hat,
        sourceImage=sourceImage,
        target_width=300,
        target_height=300,
    )
    IROI = IROI[0]

    # Convert the ROI to a PIL image
    base_image = Image.fromarray(np.uint8(IROI.cpu()[0])).convert("L")
    # Define basic transformations
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    center_crop = transforms.CenterCrop((224, 224))
    grayscale = transforms.Grayscale(num_output_channels=1)

    def process_variant(img, transform):
        img = transform(img)
        img_tensor = to_tensor(img)
        img_tensor = img_tensor.repeat(3, 1, 1)  # Repeat grayscale to 3 channels
        img_tensor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(
            img_tensor
        )
        return img_tensor.unsqueeze(0).to(
            device
        )  # Add batch dimension and move to device

    # Image 1: Grayscale + CenterCrop
    img1 = process_variant(base_image, transforms.Compose([center_crop, grayscale]))

    # Images 2 and 3: Color shift + Grayscale + CenterCrop
    color_jitter = transforms.ColorJitter(
        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
    )
    img2 = process_variant(
        base_image, transforms.Compose([color_jitter, center_crop, grayscale])
    )
    img3 = process_variant(
        base_image, transforms.Compose([color_jitter, center_crop, grayscale])
    )  # Different random color shift

    # Images 4 and 5: Rotate first, then CenterCrop + Grayscale
    rotate1 = transforms.RandomRotation(degrees=(-50, 50))
    rotate2 = transforms.RandomRotation(degrees=(-50, 50))
    img4 = process_variant(
        base_image, transforms.Compose([rotate1, center_crop, grayscale])
    )
    img5 = process_variant(
        base_image, transforms.Compose([rotate2, center_crop, grayscale])
    )

    # Image 6: Cut-out, then CenterCrop + Grayscale
    def cutout(image, size=40):
        np_image = np.array(image)
        h, w = np_image.shape[:2]
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        np_image[y : y + size, x : x + size] = 0
        return Image.fromarray(np_image)

    img6 = process_variant(
        base_image, transforms.Compose([cutout, center_crop, grayscale])
    )

    return [img1, img2, img3, img4, img5, img6]


def predictAndPreprocess(path: str):
    """
    Preprocess a single image and return the CNN-ready tensor.
    @path: path to hand image
    @returns: Preprocessed image tensor
    """
    grayTransformer = transforms.Compose(
        [
            transforms.CenterCrop((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    CNNtransformer = transforms.Compose(
        [
            transforms.Grayscale(),  # Ensure grayscale for palm-vein images
            transforms.CenterCrop((224, 224)),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),  # Shift hue and contrast
            transforms.ToTensor(),
            transforms.Lambda(
                lambda x: x.repeat(3, 1, 1)
            ),  # Repeat grayscale channel to have 3 channels
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    recognitionNetwork.eval()
    inputPIL = Image.open(path).convert("RGB")
    (PILMain, sourceImage, resizedImage) = getOriginalAndResizedInput(inputPIL)

    sourceImage = torch.stack([sourceImage.squeeze()])
    resizedImage = torch.stack([resizedImage.squeeze()])

    # get normalized coordinates
    theta_hat = getThetaHat(resizedImage, localisationNetwork)

    # get all ROIs
    IROI = sampleGrid(
        theta_hat=theta_hat,
        sourceImage=sourceImage,
        target_width=300,
        target_height=300,
    )
    IROI = IROI[0]

    # Convert the ROI to a PIL image and apply transformations for CNN
    b = Image.fromarray(np.uint8(IROI.cpu()[0])).convert("L")
    img = CNNtransformer(b)

    # Add a batch dimension and move to the correct device
    img = img.view(1, img.size(0), img.size(1), img.size(2))
    img = img.to(device)

    return img


def predictAndShow(path: str):
    """
    Single Image Prediction
    @path: path to hand image (Tongji)
    """
    grayTransformer = transforms.Compose(
        [
            transforms.CenterCrop((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    CNNtransformer = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    fig = plt.figure(figsize=(15, 15))  # specifying the overall grid size
    recognitionNetwork.eval()
    inputPIL = Image.open(path).convert("RGB")
    plt.subplot(2, 2, 1)
    plt.imshow(inputPIL)
    plt.title("Raw Input Image")
    (PILMain, sourceImage, resizedImage) = getOriginalAndResizedInput(inputPIL)
    sourceImage = torch.stack([sourceImage.squeeze()])
    resizedImage = torch.stack([resizedImage.squeeze()])
    # get normalized coordinates
    theta_hat = getThetaHat(resizedImage, localisationNetwork)
    plt.subplot(2, 2, 2)
    plt.imshow(markImage(inputPIL, theta_hat[0]))
    plt.title("Estimated Landmarks")
    # get all ROIs
    IROI = sampleGrid(
        theta_hat=theta_hat,
        sourceImage=sourceImage,
        target_width=300,
        target_height=300,
    )
    IROI = IROI[0]
    plt.subplot(2, 2, 3)
    plt.imshow((IROI.cpu()[0]), cmap="gray")
    plt.title("Extracted ROI")
    b = Image.fromarray(np.uint8(IROI.cpu()[0])).convert("L")
    img = CNNtransformer(b)
    plt.subplot(2, 2, 4)
    plt.imshow((img.cpu()[0]), cmap="gray")
    plt.imsave(path, img.cpu()[0], cmap='gray')
    plt.title("Preprocessed for CNN classification")
    img = img.view(1, img.size(0), img.size(1), img.size(2))
    img = img.to(device)
    with torch.no_grad():
        resp = recognitionNetwork.forward(img)
    m = nn.Softmax(dim=1)
    classi = m(resp)
    # set the spacing between subplots
    plt.subplots_adjust(
        left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4
    )
    plt.plot()
    return fig
