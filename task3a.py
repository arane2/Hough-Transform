import cv2
import numpy as np
import math

img_hough = cv2.imread('hough.jpg', 0)
img_canny = cv2.Canny(img_hough, 100, 250)
cv2.imwrite("Hough_EdgeDetection.jpg",img_canny)

ker_hough = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]])

def convol(img, ker):
    convol_img = np.zeros((img.shape))
    img_sub = np.ndarray((ker.shape))
    m = img.shape[0]  # m-> no. of rows in Image
    n = img.shape[1]
    p = ker.shape[0]  # p -> no. of rows in Kernel
    q = ker.shape[1]
    rows = math.floor(p / 2)  # rows - > first row
    cols = math.floor(q / 2)
    for i in range(rows, m - rows):
        for j in range(cols, n - cols):
            img_sub = img[i - rows: i + rows + 1, j - cols: j + cols + 1]
            summ = 0.0
            for k in range(p):
                for l in range(q):
                    summ = summ + img_sub[k][l] * ker[k][l]
            if summ > 1529:                                   #We do thresholding here itself
                convol_img[i][j] = 255
    return convol_img

convol_hough = convol(img_canny, ker_hough)
cv2.imwrite("Hough_ConvolThreshold.jpg",convol_hough)

rows = convol_hough.shape[0]
cols = convol_hough.shape[1]
diag_len = np.ceil(np.sqrt(rows * rows + cols * cols))

huff_matrix = np.zeros((int(diag_len), 181))

for i in range(convol_hough.shape[0]):
    for j in range(convol_hough.shape[1]):
        if convol_hough[i][j] == 255:
            for theta in range(0, 181):
                rho = int(np.round(i * np.cos(np.deg2rad(theta)) + j * np.sin(np.deg2rad(theta))))
                huff_matrix[rho, theta] += 1

max_rho = []
maxval = []
maxrho = []
max_rho = 0
maxtheta = []
max_theta = 0
maxvalue = 0
maxvalue2 = 0

for z in range(15):
    for i in range(huff_matrix.shape[0]):
        for j in range(huff_matrix.shape[1]):
            if huff_matrix[i, j] > maxvalue and z==0:
                maxvalue = huff_matrix[i, j]
                max_rho = i
                max_theta = j
            if huff_matrix[i, j] < maxvalue and huff_matrix[i, j] > maxvalue2 and z!=0:
                maxvalue2 = huff_matrix[i, j]
                max_rho = i
                max_theta = j
    if z==0:
        maxval.append(maxvalue)
    else:
        maxval.append(maxvalue2)
        maxvalue = maxvalue2
        maxvalue2 = 0
    maxrho.append(max_rho)
    maxtheta.append(max_theta)

#--------------------------
maxval_unique = []
maxrho_unique = []
maxtheta_unique = []
indexes_unique = []
sortedrho = []

sortedrho = sorted(maxrho)
indexes_unique.append(sortedrho[0])
for i in range(len(sortedrho)-1):
    if(sortedrho[i+1] - sortedrho[i] > 30):
        indexes_unique.append(sortedrho[i+1])
for i in range(len(indexes_unique)):
    for j in range(len(maxrho)):
        if(maxrho[j] == indexes_unique[i]):
            maxval_unique.append(maxval[j])
            maxrho_unique.append(maxrho[j])
            maxtheta_unique.append(maxtheta[j])

#---------------------------
vertical_lines = np.zeros(convol_hough.shape)
for z in range(6):
    for i in range(450):
        x = i
        y = math.ceil((-x / math.tan(math.radians(maxtheta_unique[z]))) + maxrho_unique[z] / math.sin(
            math.radians(maxtheta_unique[z])))
        vertical_lines[x][y] = 255
        convol_hough[x][y] = 255

#----------------------
cv2.imwrite("RedLinesDetected.jpg",vertical_lines)
cv2.imwrite("RedlinesonImage.jpg", convol_hough)