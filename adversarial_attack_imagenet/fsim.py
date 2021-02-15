# coding:utf-8
# author:liangsiyuan
# @Time :2020/5/23  下午9:38

# coding:utf-8
# author:liangsiyuan
# @Time :2020/2/2  下午3:54

import cv2
import os
import phasepack
import numpy as np
from timeit import default_timer as timer
from skimage.measure import compare_psnr, compare_ssim


def compute_fsim(im1, im2):
    """Compute the Feature Similarity Index (FSIM) between two images.
    Parameters
    ----------
    im1, im2 : ndarray
        Image.  Any dimensionality.
    Returns
    -------
    fsim : float
        The FSIM metric.
    pc_max : ndarray
        Maximum Phase Congruency Feature Map Image.
        A map combining the important structures in each image.
    References
    ----------
    Zhang, L., Zhang, L., Mou, X., & Zhang, D. (2011).
    FSIM: A feature similarity index for image quality assessment.
    IEEE Transactions on Image Processing, 20(8), 2378–2386.
    http://doi.org/10.1109/TIP.2011.2109730
    Notes
    -------
    """

    # print("Computing Feature Similarity...")
    start = timer()

    # Stability constants
    t1 = 0.85
    t2 = 160

    # First we construct Phase Congruency (PC) images.
    # PC is a dimensionless measure of the significance of local structure.
    # We rely on the phasepack library (https://github.com/alimuldal/phasepack) for this computation.
    # The parameters may vary from the implementation of Zhang et al.
    pc1 = phasepack.phasecong(im1, nscale=4, norient=4,
                              minWaveLength=6, mult=2, sigmaOnf=0.55)[0]
    pc2 = phasepack.phasecong(im2, nscale=4, norient=4,
                              minWaveLength=6, mult=2, sigmaOnf=0.55)[0]

    # Next we compute the similarity of the PC images
    s_pc = (2 * pc1 * pc2 + t1) / (pc1 ** 2 + pc2 ** 2 + t1)

    # Next we compute the Sobel gradient magnitude representation of each image in both the x and y direction.
    im1_gradient_x = cv2.Sobel(im1, cv2.CV_64F, dx=1, dy=0, ksize=-1)
    im1_gradient_y = cv2.Sobel(im1, cv2.CV_64F, dx=0, dy=1, ksize=-1)
    im2_gradient_x = cv2.Sobel(im2, cv2.CV_64F, dx=1, dy=0, ksize=-1)
    im2_gradient_y = cv2.Sobel(im2, cv2.CV_64F, dx=0, dy=1, ksize=-1)

    # These gradients are used to construct a gradient magnitude feature map for im1 and im2.
    im1_gm = np.sqrt((im1_gradient_x**2) + (im1_gradient_y**2))
    im2_gm = np.sqrt((im2_gradient_x**2) + (im2_gradient_y**2))

    # Now we have two feature maps: Phase Congruency and Gradient Magnitude (GM)
    # We will now compute the similarity of the GM maps.
    s_gm = (2 * im1_gm * im2_gm + t2) / (im1_gm ** 2 + im2_gm ** 2 + t2)

    # We simply combine the GM and PC similarity to compute the total similarity
    if len(s_gm.shape) == 3:
        s_pc = s_pc[:, :, np.newaxis]
        s_pc = np.repeat(s_pc, 3, axis=2)
    # s_gm = np.sum(s_gm, axis=2)
    s_total = s_pc * s_gm

    # However, different locations have different contributions to the perception of image similarity.
    # For example, edges are more important than smooth areas, so high PC values indicate important structures.
    # We then weight the importance using the maximum values of the PC image pair.
    pc_max = np.maximum(pc1, pc2)
    if len(s_gm.shape) == 3:
        pc_max = pc_max[:, :, np.newaxis]
        pc_max = np.repeat(pc_max, 3, axis=2)
    fsim = np.sum(s_total * pc_max) / np.sum(pc_max)
    end = timer()
    # print("Computing Feature Similarity...Complete. Elapsed Time: [s] " + str(end - start))

    return fsim, pc_max

if __name__ == '__main__':
    a_path = 'adv_data/0.0/'
    b_path = 'adv_data/0.5/'
    imgs = os.listdir(a_path)
    ssim_total = 0
    psnr_total = 0
    for img in imgs:
        im1 = cv2.imread(a_path+img, cv2.IMREAD_COLOR)
        # im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
        im2 = cv2.imread(b_path+img, cv2.IMREAD_COLOR)
        # im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
        ssim_total = ssim_total + compare_ssim(im1, im2, data_range=255, multichannel=True)
        psnr_total = psnr_total + compare_psnr(im2, im1)

    ssim_total = ssim_total / len(imgs)
    psnr_total = psnr_total / len(imgs)

    print('ssim: '+ str(ssim_total))
    print('psnr: '+ str(psnr_total))