import cv2
import numpy as np
import math

img = cv2.imread('noise.jpg', 0)
cv2.imwrite('noisee.jpg', img)
print(img.shape)

a = 3
b = 3
ker = np.ones((a, b))
kernellen = a * b


def erosion(image, kernel):
    erosimg = np.zeros(image.shape)
    rows = math.floor(kernel.shape[0] / 2)
    cols = math.floor(kernel.shape[1] / 2)
    for i in range(rows, image.shape[0] - rows):
        for j in range(cols, image.shape[1] - rows):
            temp = image[i - rows: i + rows + 1, j - cols: j + cols + 1]
            count = 0
            for k in range(kernel.shape[0]):
                for l in range(kernel.shape[1]):
                    if (temp[k][l] == 255):
                        count += 1
            if (count == kernellen):
                erosimg[i][j] = 255
    return erosimg


def dilation(image, kernel):
    erosimg = np.zeros(image.shape)
    rows = math.floor(kernel.shape[0] / 2)
    cols = math.floor(kernel.shape[1] / 2)
    for i in range(rows, image.shape[0] - rows):
        for j in range(cols, image.shape[1] - rows):
            temp = image[i - rows: i + rows + 1, j - cols: j + cols + 1]
            count = 0
            for k in range(kernel.shape[0]):
                for l in range(kernel.shape[1]):
                    if (temp[k][l] == 255):
                        count = 1
            if (count == 1):
                erosimg[i][j] = 255
    return erosimg


OTC = erosion(dilation(dilation(erosion(img, ker), ker), ker), ker)
cv2.imwrite('res_noise1.jpg', OTC)

CTO = dilation(erosion(erosion(dilation(img, ker), ker), ker), ker)
cv2.imwrite('res_noise2.jpg', CTO)

EOTC = erosion(OTC, ker)
ECTO = erosion(CTO, ker)
BoundaryOTC = OTC - EOTC
BoundaryCTO = CTO - ECTO

cv2.imwrite('res_bound1.jpg', BoundaryOTC)
cv2.imwrite('res_bound2.jpg', BoundaryCTO)


