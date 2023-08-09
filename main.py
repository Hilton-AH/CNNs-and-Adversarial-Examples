#Please place any imports here.
# BEGIN IMPORTS

import numpy as np
import cv2
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, datasets

import scipy.ndimage as rotate

import random
from PIL import Image

# END IMPORTS

#########################################################
###              BASELINE MODEL
########################################################

class AnimalBaselineNet(nn.Module):
    def __init__(self, num_classes=16):
        super(AnimalBaselineNet, self).__init__()
        # TODO: Define layers of model architecture
        # TODO-BLOCK-BEGIN
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 6, kernel_size = 3, stride = 2, padding = 1)
        self.nonlinearity1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 12, kernel_size = 3, stride = 2, padding = 1)
        self.nonlinearity2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels = 12, out_channels = 24, kernel_size = 3, stride = 2, padding = 1)
        self.nonlinearity3 = nn.ReLU()
        self.fc = nn.Linear(in_features = 1536, out_features = 128)
        self.nonlinearity4 = nn.ReLU()
        self.cls = nn.Linear(in_features = 128, out_features = 16)
        

        # TODO-BLOCK-END

    def forward(self, x):
        x = x.contiguous().view(-1, 3, 64, 64).float()

        # TODO: Define forward pass
        # TODO-BLOCK-BEGIN
        x = self.conv1(x)
        x = self.nonlinearity1(x)
        x = self.conv2(x)
        x = self.nonlinearity2(x)
        x = self.conv3(x)
        x = self.nonlinearity3(x)
        x = x.view(-1, 1536)
        x = self.fc(x)
        x = self.nonlinearity4(x)
        x = self.cls(x)

        # TODO-BLOCK-END
        return x

def model_train(net, inputs, labels, criterion, optimizer):
    """
    Will be used to train baseline and student models.

    Inputs:
        net        network used to train
        inputs     (torch Tensor) batch of input images to be passed
                   through network
        labels     (torch Tensor) ground truth labels for each image
                   in inputs
        criterion  loss function
        optimizer  optimizer for network, used in backward pass

    Returns:
        running_loss    (float) loss from this batch of images
        num_correct     (torch Tensor, size 1) number of inputs
                        in this batch predicted correctly
        total_images    (float or int) total number of images in this batch

    Hint: Don't forget to zero out the gradient of the network before the backward pass. We do this before
    each backward pass as PyTorch accumulates the gradients on subsequent backward passes. This is useful
    in certain applications but not for our network.
    """
    # TODO: Foward pass
    # TODO-BLOCK-BEGIN
    outputs = net(inputs)
    loss = criterion(outputs, labels.squeeze())
    _, preds = torch.max(outputs, 1)
    # TODO-BLOCK-END

    # TODO: Backward pass
    # TODO-BLOCK-BEGIN
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # TODO-BLOCK-END
    
    total_images = labels.data.numpy().size
    num_correct  = torch.sum(preds == labels.data.reshape(-1))
    running_loss = loss.item()

    return running_loss, num_correct, total_images

#########################################################
###               DATA AUGMENTATION
#########################################################

class Shift(object):
    """
  Shifts input image by random x amount between [-max_shift, max_shift]
    and separate random y amount between [-max_shift, max_shift]. A positive
    shift in the x- and y- direction corresponds to shifting the image right
    and downwards, respectively.

    Inputs:
        max_shift  float; maximum magnitude amount to shift image in x and y directions.
    """
    def __init__(self, max_shift=10):
        self.max_shift = max_shift

    def __call__(self, image):
        """
        Inputs:
            image         3 x H x W image as torch Tensor

        Returns:
            shift_image   3 x H x W image as torch Tensor, shifted by random x
                          and random y amount, each amount between [-max_shift, max_shift].
                          Pixels outside original image boundary set to 0 (black).
        """
        image = image.numpy()
        _, H, W = image.shape
        # TODO: Shift image
        # TODO-BLOCK-BEGIN
        tx = random.randint(-self.max_shift, self.max_shift)
        ty = random.randint(-self.max_shift, self.max_shift)
        M = np.float32([[1, 0, tx],
                    [0, 1, ty]])
        image[0] = cv2.warpAffine(image[0], M, (H, W))
        image[1] = cv2.warpAffine(image[1], M, (H, W))
        image[2] = cv2.warpAffine(image[2], M, (H, W))
        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__

class Contrast(object):
    """
    Randomly adjusts the contrast of an image. Uniformly select a contrast factor from
    [min_contrast, max_contrast]. Setting the contrast to 0 should set the intensity of all pixels to the
    mean intensity of the original image while a contrast of 1 returns the original image.

    Inputs:
        min_contrast    non-negative float; minimum magnitude to set contrast
        max_contrast    non-negative float; maximum magnitude to set contrast

    Returns:
        image        3 x H x W torch Tensor of image, with random contrast
                     adjustment
    """

    def __init__(self, min_contrast=0.3, max_contrast=1.0):
        self.min_contrast = min_contrast
        self.max_contrast = max_contrast

    def __call__(self, image):
        """
        Inputs:
            image         3 x H x W image as torch Tensor

        Returns:
            shift_image   3 x H x W torch Tensor of image, with random contrast
                          adjustment
        """
        image = image.numpy()
        _, H, W = image.shape

        # TODO: Change image contrast
        # TODO-BLOCK-BEGIN

        c = random.uniform(self.min_contrast, self.max_contrast)
        
        mean_img_c0 = np.mean(image[0])
        mean_img_c1 = np.mean(image[1])
        mean_img_c2 = np.mean(image[2])
        
        adj_image = np.zeros((3, H, W))
        
        adj_image[0] = mean_img_c0 * (1 - c) + image[0] * c
        adj_image[1] = mean_img_c1 * (1 - c) + image[1] * c
        adj_image[2] = mean_img_c2 * (1 - c) + image[2] * c

        # TODO-BLOCK-END

        return torch.Tensor(adj_image)

    def __repr__(self):
        return self.__class__.__name__

class Rotate(object):
    """
    Rotates input image by random angle within [-max_angle, max_angle]. Positive angle corresponds to
    counter-clockwise rotation

    Inputs:
        max_angle  maximum magnitude of angle rotation, in degrees


    """
    def __init__(self, max_angle=10):
        self.max_angle = max_angle

    def __call__(self, image):
        """
        Inputs:
            image           image as torch Tensor

        Returns:
            rotated_image   image as torch Tensor; rotated by random angle
                            between [-max_angle, max_angle].
                            Pixels outside original image boundary set to 0 (black).
        """
        image = image.numpy()
        _, H, W  = image.shape

        # TODO: Rotate image
        # TODO-BLOCK-BEGIN
        angle = random.uniform(-self.max_angle, self.max_angle)
        M = cv2.getRotationMatrix2D((W//2, H//2), angle, 1.0)
        image[0] = cv2.warpAffine(image[0], M, (H, W))
        image[1] = cv2.warpAffine(image[1], M, (H, W))
        image[2] = cv2.warpAffine(image[2], M, (H, W))
        # TODO-BLOCK-END

        return torch.Tensor(image)

    def __repr__(self):
        return self.__class__.__name__

class HorizontalFlip(object):
    """
    Randomly flips image horizontally.

    Inputs:
        p          float in range [0,1]; probability that image should
                   be randomly rotated
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image):
        """
        Inputs:
            image           image as torch Tensor

        Returns:
            flipped_image   image as torch Tensor flipped horizontally with
                            probability p, original image otherwise.
        """
        image = image.numpy()
        _, H, W = image.shape

        # TODO: Flip image
        # TODO-BLOCK-BEGIN
        choice = random.uniform(0, 1)
        if choice <= self.p:
            image = np.flip(image, axis=2)
        

        # TODO-BLOCK-END

        return torch.Tensor(image.copy())

    def __repr__(self):
        return self.__class__.__name__

#########################################################
###             STUDENT MODEL
#########################################################

def get_student_settings(net):
    """
    Return transform, batch size, epochs, criterion and
    optimizer to be used for training.
    """
    dataset_means = [123./255., 116./255.,  97./255.]
    dataset_stds  = [ 54./255.,  53./255.,  52./255.]

    # TODO: Create data transform pipeline for your model
    # transforms.ToPILImage() must be first, followed by transforms.ToTensor()
    # TODO-BLOCK-BEGIN

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(64, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(dataset_means, dataset_stds)
    ])

    # TODO-BLOCK-END

    # TODO: Settings for dataloader and training. These settings
    # will be useful for training your model.
    # TODO-BLOCK-BEGIN

    batch_size = 100
    epochs = 10

    # TODO-BLOCK-END

    # TODO: epochs, criterion and optimizer
    # TODO-BLOCK-BEGIN

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    # TODO-BLOCK-END

    return transform, batch_size, epochs, criterion, optimizer

class AnimalStudentNet(nn.Module):
    def __init__(self, num_classes=16):
        super(AnimalStudentNet, self).__init__()
        # TODO: Define layers of model architecture
        # TODO-BLOCK-BEGIN

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.25)

        # TODO-BLOCK-END

    def forward(self, x):
        x = x.contiguous().view(-1, 3, 64, 64).float()

        # TODO: Define forward pass
        # TODO-BLOCK-BEGIN

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # TODO-BLOCK-END
        return x

#########################################################
###             ADVERSARIAL IMAGES
#########################################################

def get_adversarial(img, output, label, net, criterion, epsilon):
    """
    Generates adversarial image by adding a small epsilon
    to each pixel, following the sign of the gradient.

    Inputs:
        img        (torch Tensor) image propagated through network
        output     (torch Tensor) output from forward pass of image
                   through network
        label      (torch Tensor) true label of img
        net        image classification model
        criterion  loss function to be used
        epsilon    (float) perturbation value for each pixel

    Outputs:
        perturbed_img   (torch Tensor, same dimensions as img)
                        adversarial image, clamped such that all values
                        are between [0,1]
                        (Clamp: all values < 0 set to 0, all > 1 set to 1)
        noise           (torch Tensor, same dimensions as img)
                        matrix of noise that was added element-wise to image
                        (i.e. difference between adversarial and original image)

    Hint: After the backward pass, the gradient for a parameter p of the network can be accessed using p.grad
    """

    # TODO: Define forward pass
    # TODO-BLOCK-BEGIN

    img.requires_grad = True

    # Forward pass and calculate the loss
    output = net(img)
    loss = criterion(output, label)
    
    # Zero the gradients and perform a backward pass
    net.zero_grad()
    loss.backward()
    
    # Calculate the perturbation matrix
    gradient_sign = img.grad.sign()
    perturbation = epsilon * gradient_sign
    
    # Create the perturbed image
    perturbed_image = img + perturbation
    
    # Clamp the perturbed image
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    # Calculate the noise
    noise = perturbed_image - img

    # TODO-BLOCK-END

    return perturbed_image, noise

