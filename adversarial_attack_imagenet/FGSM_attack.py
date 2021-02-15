# coding:utf-8
# author:liangsiyuan
# @Time :2020/5/23  下午9:36
import torchvision.models as model
import eagerpy as ep
from models.pytorch import PyTorchModel
from utils import accuracy, samples
from attacks import FGSM, LinfPGD, L2DeepFoolAttack

if __name__ == '__main__':
    model = model.resnet152(pretrained=True).eval()
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    # preprocessing = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], axis=-3)
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
    total_robust_acc = 0
    # for index in range(0, 15, 5):
    #     images, labels = ep.astensors(*samples(fmodel, dataset="imagenet", batchsize=5, index=index))
    #     print(accuracy(fmodel, images, labels, False))
    #
    #     attack = FGSM()
    #     epsilons = [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0]
    #     advs, _, success = attack(fmodel, images, labels, epsilons=epsilons)
    #
    #     robust_accuracy = 1 - success.float32().mean(axis=-1)
    #     total_robust_acc = total_robust_acc + robust_accuracy
    #     # for eps, acc in zip(epsilons, robust_accuracy):
    #     #     print(eps, acc.item())
    # for eps, acc in zip(epsilons, total_robust_acc/4):
    #     print(eps, acc.item())

    images, labels = ep.astensors(*samples(fmodel, dataset="imagenet", batchsize=20))
    print(accuracy(fmodel, images, labels, False))

    # attack = LinfPGD()
    # attack = L2DeepFoolAttack()
    attack = FGSM()
    epsilons = [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0]
    advs, _, success = attack(fmodel, images, labels, epsilons=epsilons)

    robust_accuracy = 1 - success.float32().mean(axis=-1)
    total_robust_acc = total_robust_acc + robust_accuracy
    for eps, acc in zip(epsilons, robust_accuracy):
        print(eps, acc.item())