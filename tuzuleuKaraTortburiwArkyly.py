import cv2
import numpy as np
from imutils import contours

original_image = cv2.imread("images/not_empty/Scan_0002.jpg")

def show_small_image(image_description, image):
    small = cv2.resize(image, (0, 0), fx=0.29, fy=0.29)
    cv2.imshow(image_description, small)
    cv2.waitKey(0)

kernel = np.ones((5,5),np.float32)/25
dst = cv2.filter2D(original_image,-1,kernel)

blur = cv2.blur(original_image,(5,5))

blur2 = cv2.GaussianBlur(original_image,(5,5),0)

median = cv2.medianBlur(original_image, 5)

def finding_the_black_squares(image):

    black_squares = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 20, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C | cv2.THRESH_OTSU)[1]
    im2, cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = contours.sort_contours(cnts, "top-to-bottom")[0]
    # cnts = contours.sort_contours(cnts, "left-to-right")[0]
    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            # cv2.drawContours(image, c, -1, (10, 200, 50), 50)
            # show_small_image(image)
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)

            if len(approx) == 4 and cv2.contourArea(c) > 3500 and cv2.contourArea(c) < 5000:
                # print(approx)
                # approx = contours.sort_contours(approx, "top-to-bottom")[0]

                black_squares.append([approx])
                cv2.drawContours(image, approx, -1, (0,0,255), 15)
                show_small_image("Black squares: ", image)
    # print(black_squares)
    return black_squares

black_squares = finding_the_black_squares(original_image)

contours_list = []
for i in range(len(black_squares)):
    # print(black_squares[i][0])
    for j in range(len(black_squares[i][0])):
        contours_list.append([black_squares[i][0][j][0][0], black_squares[i][0][j][0][1]])
print("Contours List: ", contours_list)


# for i in range(len(contours_list)):
sorted_contours_list_by_X = sorted(contours_list, key=lambda row: row[0])
contours_list = sorted_contours_list_by_X
print(contours_list)
# sorted_contours_list_by_Y = sorted(sorted_contours_list_by_X, key=lambda row: row[1])
# for i in range(len(contours_list)):
def takeXPoints(contours_list):
    x_points = []
    for i in range(len(contours_list)):
        x_point = contours_list[i][0]
        x_points.append(x_point)
    return x_points


def takeYPoints(contours_list):
    y_points = []
    for i in range(len(contours_list)):
        y_point = contours_list[i][1]
        y_points.append(y_point)
    return y_points


def finding_theMaxMinOrMinMaxPoints(contours_list, max_x, min_x, max_y, min_y):
    maxX_minY= None
    maxY_minX = None
    for i in range(len(contours_list)):
        if min_y <= contours_list[i][1] <= min_y + 60 and max_x - 60 <= contours_list[i][0] <= max_x:
            maxX_minY = contours_list[i]
            # print(contours_list[i])
            # cv2.circle(original_image, (contours_list[i][0], contours_list[i][1]), 15, (0, 255, 0),-1)
            #     # cv2.drawContours(original_image, black_squares[i][j], -1, (0, 255, 0), 15)
            # show_small_image("Black squares: ", original_image)
        if min_x <= contours_list[i][1] <= min_x + 60 and max_y - 60 <= contours_list[i][0] <= max_y:
            maxY_minX = contours_list[i]
            # print(contours_list[i])
            # cv2.circle(original_image, (contours_list[i][0], contours_list[i][1]), 15, (0, 255, 0), -1)
            #     # cv2.drawContours(original_image, black_squares[i][j], -1, (0, 255, 0), 15)
            # show_small_image("Black squares: ", original_image)
    return maxX_minY, maxY_minX

def addManuallyOnePoint(original_image):
    fourth_point = None
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 20, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C | cv2.THRESH_OTSU)[1]
    im2, cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # (x_black_square, y_black_square, w_black_square, h_black_square) = cv2.boundingRect(black_squares[0])

    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
            (x, y, w, h) = cv2.boundingRect(c)

            if len(approx) == 4 and cv2.contourArea(c) > 10000 and y > original_image.shape[0] - 1500:
                cnts = contours.sort_contours(approx, method="left-to-right")[0]
                cnts = contours.sort_contours(approx, method="top-to-bottom")[0]
                for c in cnts:
                    perimeter = cv2.arcLength(c, True)
                    approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
                    (x, y, w, h) = cv2.boundingRect(c)
                    if y > original_image.shape[0] - 100 and x < 150:
                        fourth_point = [x, y]
                        cv2.circle(original_image, (x, y), 15,
                                   (200, 0, 200), -1)
                        show_small_image("Black squares: ", original_image)
    print("Fourth point: ", fourth_point)
    return fourth_point

x_points = takeXPoints(contours_list)
y_points = takeYPoints(contours_list)
max_x = max(x_points)
min_x = min(x_points)
max_y = max(y_points)
min_y = min(y_points)

sumOfXYCoordinates = []
for i in range(len(contours_list)):
    Xi = contours_list[i][0]
    Yi = contours_list[i][1]
    sumOfXY = Xi + Yi
    sumOfXYCoordinates.append(sumOfXY)

print(sumOfXYCoordinates)
max_sum = max(sumOfXYCoordinates)
min_sum = min(sumOfXYCoordinates)

maxX_maxY = None
minX_minY = None

for i in range(len(contours_list)):
    Xi = contours_list[i][0]
    Yi = contours_list[i][1]
    sumOfXY = Xi + Yi
    if sumOfXYCoordinates[i] == max_sum:
        maxX_maxY = contours_list[i]
        print(contours_list[i])
    if sumOfXYCoordinates[i] == min_sum:
        minX_minY = contours_list[i]


cv2.circle(original_image, (minX_minY[0], minX_minY[1]), 15, (255, 0, 0),-1)
show_small_image("Black squares: ", original_image)



fourth_point = addManuallyOnePoint(original_image)


maxX_minY, maxY_minX = finding_theMaxMinOrMinMaxPoints(contours_list, max_x, min_x, max_y, min_y)
four_points = [minX_minY, maxX_minY, maxX_maxY, fourth_point]
print("Four points: ", four_points)


for i in range(len(four_points)):
    cv2.circle(original_image, (four_points[i][0], four_points[i][1]), 15, (0, 255, 0),-1)
    show_small_image("Black squares: ", original_image)


print("Maximum Y, Minimum X: ", maxY_minX)
print("Maximum X, Minimum Y: ", maxX_minY)
print("Maximum X: ", max_x)
print("Minimum X: ", min_x)
print("Maximum Y: ", max_y)
print("Minimum Y: ", min_y)
print("Maximum point (X,Y): ", maxX_maxY)
print("Minimum point (X,Y): ", minX_minY)

# print(sorted_contours_list_by_X)
    # for i in range(len(contours_list)):
    #     print(contours_list[i][0])
    #     if i == len(contours_list):
    #         break
    #     if contours_list[i][0] > contours_list[i+1][0]:
    #         max_x = contours_list[i][0]
    #         min_x = contours_list[i+1][0]
    # else:
    #     min_x = contours_list[i][0]
# print("Maximum X: ", max_x)
# print("Minimum X: ", min_x)

# numberOfRows = 4
# sorted_black_squares = []
# for (row, i) in enumerate(np.arange(0, len(contours_list), numberOfRows)):
#     # print("Number of rows: ", numberOfRows)
#     sorted_black_squares.append(contours_list[i:i+numberOfRows])
#     # print(row, i)
# print(sorted_black_squares)
#
# for i in range(len(sorted_black_squares)):
#     sorted_per_square = sorted(sorted_black_squares[i], key=lambda row: row[0])
#     sorted_black_squares[i] = sorted_per_square
#
# print("Sorted by X black squares: ", sorted_black_squares)
#
# for i in range(len(sorted_black_squares)):
#     print(len(sorted_black_squares)-1)
#     if i == len(sorted_black_squares[i])-1:
#         break
#     max_x_y = sorted_black_squares[i][-1]
#     max_x2_y2 = sorted_black_squares[i+1][-1]
#     # print(max_x_y)
#     if  max_x_y[0] > max_x2_y2[0] and max_x_y[1] > max_x2_y2[1]:
#         print(max_x_y[0])
# print(sorted_black_squares)
# for i in range(len(sorted_black_squares)):
#     for j in range(len(sorted_black_squares[i])):
#         if j == len(sorted_black_squares[i][0]):
#             break
#         if sorted_black_squares[i][j][0]>sorted_black_squares[i][j+1][0]:
#             print(sorted_black_squares[i][j][0], sorted_black_squares[i][j+1][0])
#             max_x = sorted_black_squares[i][j][0]

# print(sorted_black_squares)
# # sorted_black_squares_by_x = sort_the_circles_ByX(per_column)
#
# # sorted_black_square_contours = sorted(contours_list, key=lambda row: row[0])
# # print(sorted_black_square_contours)
#
# for i in sorted_black_squares:
#     for j in range(len(i)):
#         cv2.circle(original_image, (i[j][0], i[j][1]), 15, (0, 255, 0), -1)
#         show_small_image("Black squares: ", original_image)

    # for j in range(len(black_squares[i])):
    #     cv2.circle(original_image, (black_squares[i][j][2][0][0], black_squares[i][j][2][0][1]), 15, (0, 255, 0), -1)
    #     # cv2.drawContours(original_image, black_squares[i][j], -1, (0, 255, 0), 15)
    #     show_small_image("Black squares: ", original_image)
    #     print(black_squares[i][j][0][0])
        # print(sorted(black_squares[i][j][0], key=lambda a_entry: a_entry[1]))

# circles = sorted(circles[0], key=lambda row: row[0])
# show_small_image("Original image: ", original_image)
# show_small_image("Smoothing: ", dst)
# show_small_image("Blur: ", blur)
# show_small_image("Gaussian blur: ", blur2)
# show_small_image("Median blur: ", median)
