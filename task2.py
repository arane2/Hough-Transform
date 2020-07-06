import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

kernel = np.array([[1, -1, 1], [-1, 4, -1], [-1, -1, -1]])
img_point = cv2.imread('turbine-blade.jpg', 0)


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
            if summ >329 :                                     #We do thresholding here, in convol itself
                #if we take threshold 312 we get 2 points
                print("The Intensity and the coordinates of The Point detected is",summ, i, j)
                convol_img[i][j] = 255


    return (convol_img)

convoluted_img = convol(img_point, kernel)
cv2.imwrite('PorosityDetected.jpg', convoluted_img)

font                   = cv2.FONT_HERSHEY_SIMPLEX
coordinates = (445,250)
fontScale              = 0.4
fontColor              = (255,255,255)
lineType               = 2

cv2.putText(convoluted_img,'[250, 445]',
    coordinates,
    font,
    fontScale,
    fontColor,
    lineType)

cv2.imwrite("PorosityLabelled.jpg", convoluted_img)


#----------------Task 2b-----------------------------------------------------
# import cv2
# import numpy as np
# import math
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

img_seg = cv2.imread('segment.jpg', 0)
seg_threshold = np.zeros((img_seg.shape))


histogram = np.zeros(256)
for i in range(img_seg.shape[0]):
    for j in range(img_seg.shape[1]):
        if img_seg[i][j]<256:
            histogram[img_seg[i][j]] += 1

x,y = list(range(1, 256)), histogram[1:]
plt.title("Histogram of Intensity")
plt.ylabel('#')
plt.xlabel("Intensity Level")
plt.plot(x,y)
plt.savefig('Histogram.jpg')




for i in range(img_seg.shape[0]):
    for j in range(img_seg.shape[1]):
        if img_seg[i][j] > 204:
            seg_threshold[i][j] = 255

cv2.imwrite('SegmentedImage.jpg', seg_threshold)
i_coordinates = []  # all values of distinct i saved in list set wise
j_coordinates = []

for i in range(img_seg.shape[0]):
    for j in range(img_seg.shape[1]):
        if seg_threshold[i][j] == 255:
            j_coordinates.append(j)  # j_coordinates --> All values of j saved
j_coor = list(set(j_coordinates))  # j_coor --> Distinct values of J coordinates saved
j_coorspaced = []  # j_coorspaced --> SAve the sets of J coordinates
j_coorspaced.append(j_coor[0])
for i in range(len(j_coor) - 1):
    if j_coor[i + 1] - j_coor[i] > 10:
        j_coorspaced.append(j_coor[i])
        j_coorspaced.append(j_coor[i + 1])
j_coorspaced.append(j_coor[len(j_coor) - 1])


def icoor(seg_threshold, k, l):
    i_funcoor = []  # i_funcoor-->to store all the i points and are returned later but
    for i in range(img_seg.shape[0]):  # while returning only sets of distinct values are returned
        for j in range(k, l + 1):
            if seg_threshold[i][j] == 255:
                i_funcoor.append(i)
    return list(set(i_funcoor))


for i in range(math.floor(len(j_coorspaced) / 2)):
    i_coordinates.append(icoor(seg_threshold, j_coorspaced[2 * i], j_coorspaced[2 * i + 1]))
    # sets of distinct values of i are appended in i_coordinates

i_coorspaced = []  # to store the top and bottom of segments
for i in range(len(i_coordinates)):
    i_coorspaced.append(min(i_coordinates[i]))
    i_coorspaced.append(max(i_coordinates[i]))

print("Row Coordinates of the Boxes",i_coorspaced)
print("Column Coordinates of the Boxes",j_coorspaced)
cv2.rectangle(seg_threshold, (164, 127), (199, 162), (255, 0, 0), 2)
cv2.rectangle(seg_threshold, (255, 79), (300, 202), (255, 0, 0), 2)
cv2.rectangle(seg_threshold, (337, 27), (362, 284), (255, 0, 0), 2)
cv2.rectangle(seg_threshold, (390, 44), (420, 249), (255, 0, 0), 2)
cv2.imwrite("SegmentsWithBoundingBox.jpg", seg_threshold)