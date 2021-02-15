from typing import Optional, Tuple, Any
import eagerpy as ep
import warnings
import os
import numpy as np
import scipy.misc
import imageio
import matplotlib.pyplot as plt
import torch
import cv2
from models.types import Bounds
from models.base import Model

def accuracy(fmodel: Model, inputs: Any, labels: Any, vis: Any) -> float:
    inputs_, labels_ = ep.astensors(inputs, labels)
    del inputs, labels

    predictions = fmodel(inputs_).argmax(axis=-1)
    if vis:
        return predictions
    accuracy = (predictions == labels_).float32().mean()
    return accuracy.item()


def samples(
    fmodel: Model,
    dataset: str = "imagenet",
    index: int = 0,
    batchsize: int = 1,
    shape: Tuple[int, int] = (224, 224),
    data_format: Optional[str] = None,
    bounds: Optional[Bounds] = None,
    vis=False
) -> Any:
    if hasattr(fmodel, "data_format"):
        if data_format is None:
            data_format = fmodel.data_format  # type: ignore
        elif data_format != fmodel.data_format:  # type: ignore
            raise ValueError(
                f"data_format ({data_format}) does not match model.data_format ({fmodel.data_format})"  # type: ignore
            )
    elif data_format is None:
        raise ValueError(
            "data_format could not be inferred, please specify it explicitly"
        )

    if bounds is None:
        bounds = fmodel.bounds
    if vis:
        images, labels, file_names = _samples(
            dataset=dataset,
            index=index,
            batchsize=batchsize,
            shape=shape,
            data_format=data_format,
            bounds=bounds,
            vis=vis
        )
    else:
        images, labels = _samples(
            dataset=dataset,
            index=index,
            batchsize=batchsize,
            shape=shape,
            data_format=data_format,
            bounds=bounds,
        )

    if hasattr(fmodel, "dummy") and fmodel.dummy is not None:  # type: ignore
        images = ep.from_numpy(fmodel.dummy, images).raw  # type: ignore
        labels = ep.from_numpy(fmodel.dummy, labels).raw  # type: ignore
    else:
        warnings.warn(f"unknown model type {type(fmodel)}, returning NumPy arrays")
    if vis:
        return images, labels, file_names
    return images, labels


def _samples(
    dataset: str,
    index: int,
    batchsize: int,
    shape: Tuple[int, int],
    data_format: str,
    bounds: Bounds,
    vis=False
) -> Tuple[Any, Any]:
    # TODO: this was copied from foolbox v2

    from PIL import Image

    images, labels, file_names = [], [], []
    basepath = os.path.dirname(__file__)
    samplepath = os.path.join(basepath, "data")
    files = os.listdir(samplepath)

    if batchsize > 20:
        warnings.warn(
            "samples() has only 20 samples and repeats itself if batchsize > 20"
        )

    for idx in range(index, index + batchsize):
        i = idx % 20

        # get filename and label
        file = [n for n in files if f"{dataset}_{i:02d}_" in n][0]
        label = int(file.split(".")[0].split("_")[-1])

        # open file
        path = os.path.join(samplepath, file)
        image = Image.open(path)

        if dataset == "imagenet":
            image = image.resize(shape)

        image = np.asarray(image, dtype=np.float32)

        if image.ndim == 2:
            image = image[..., np.newaxis]

        assert image.ndim == 3

        if data_format == "channels_first":
            image = np.transpose(image, (2, 0, 1))

        images.append(image)
        labels.append(label)
        file_names.append(file)

    images_ = np.stack(images)
    labels_ = np.array(labels)
    file_names_ = np.array(file_names)

    if bounds != (0, 255):
        images_ = images_ / 255 * (bounds[1] - bounds[0]) + bounds[0]
    if vis:
        return images_, labels_, file_names_
    return images_, labels_


def tensor2im(input_image, imtype=np.uint8, cls=None):
    """"将tensor的数据类型转成numpy类型，并反归一化.

    Parameters:
        input_image (tensor) --  输入的图像tensor数组
        imtype (type)        --  转换后的numpy的数据类型
    """
    mean = [0.485,0.456,0.406] #dataLoader中设置的mean参数
    std = [0.229,0.224,0.225]  #dataLoader中设置的std参数
    # mean = [0.5,0.5,0.5] #dataLoader中设置的mean参数
    # std = [0.5,0.5,0.5]  #dataLoader中设置的std参数
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor): #如果传入的图片类型为torch.Tensor，则读取其数据进行下面的处理
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        # for i in range(len(mean)): #反标准化
        #     image_numpy[i] = image_numpy[i] * std[i] + mean[i]
        image_numpy = image_numpy.clip(min=0, max=1) * 255 #反ToTensor(),从[0,1]转为[0,255]
        image_numpy = np.transpose(image_numpy, (1, 2, 0))  # 从(channels, height, width)变为(height, width, channels)
    else:  # 如果传入的是numpy数组,则不做处理
        image_numpy = input_image
    image_numpy = image_numpy.astype(imtype)
    return image_numpy


def save_img(im, path, size, cls=None):
    """im可是没经过任何处理的tensor类型的数据,将数据存储到path中

    Parameters:
        im (tensor) --  输入的图像tensor数组
        path (str)  --  图像寻出的路径
        size (list/tuple)  --  图像合并的高宽(heigth, width)
    """
    scipy.misc.imsave(path, merge(im, size, cls)) #将合并后的图保存到相应path中
    img_ori = cv2.imread(path)
    cv2.putText(img_ori, cls, (5, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.imwrite(path, img_ori)


def merge(images, size, cls=None):
    """
    将batch size张图像合成一张大图，一行有size张图
    :param images: 输入的图像tensor数组,shape = (batch_size, channels, height, width)
    :param size: 合并的高宽(heigth, width)
    :return: 合并后的图
    """
    h, w = images[0].shape[1], images[0].shape[1]
    if (images[0].shape[0] in (3,4)): # 彩色图像
        c = images[0].shape[0]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            image = tensor2im(image, cls=cls)
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1: # 灰度图像
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            image = tensor2im(image)
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')