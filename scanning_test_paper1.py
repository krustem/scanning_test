#!/usr/bin/env python
# -*- coding: utf-8 -*-


from imutils import contours
from imutils.perspective import four_point_transform

import cv2
import numpy as np
import imutils
import xlsxwriter

def show_small_image(image_description, image):
    small = cv2.resize(image, (0, 0), fx=0.29, fy=0.29)
    cv2.imshow(image_description, small)
    cv2.waitKey(0)


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


def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def check_for_four_points(black_squares, original_image):
    four_points = None
    if len(black_squares) == 4:
        print("There are four black squares!")

        contours_list = []
        for i in range(len(black_squares)):
            # print(black_squares[i][0])
            for j in range(len(black_squares[i][0])):
                contours_list.append([black_squares[i][0][j][0][0], black_squares[i][0][j][0][1]])

        print("Contours List: ", contours_list)

        sorted_contours_list_by_X = sorted(contours_list, key=lambda row: row[0])
        contours_list = sorted_contours_list_by_X
        print("Sorted contours list:", contours_list)
        maxX_maxY, minX_minY, max_x, min_x, max_y, min_y = findingMaxAndMinPoints(contours_list)
        maxX_minY, maxY_minX = finding_theMaxMinOrMinMaxPoints(contours_list, max_x, min_x, max_y, min_y)

        four_points = [minX_minY, maxX_minY, maxX_maxY, maxY_minX]

        print(four_points)

        for i in range(len(four_points)):
            cv2.circle(original_image, (four_points[i][0], four_points[i][1]), 15, (0, 255, 0), -1)
            show_small_image("Black squares: ", original_image)

        print("First point: ", minX_minY)
        print("Second point: ", maxX_minY)
        print("Third point: ", maxX_maxY)
        print("Fourth point: ", maxY_minX)

    else:

        fourth_point = addManuallyOnePoint(original_image)

        contours_list = []
        for i in range(len(black_squares)):
            # print(black_squares[i][0])
            for j in range(len(black_squares[i][0])):
                contours_list.append([black_squares[i][0][j][0][0], black_squares[i][0][j][0][1]])
        print("Contours List: ", contours_list)

        sorted_contours_list_by_X = sorted(contours_list, key=lambda row: row[0])
        contours_list = sorted_contours_list_by_X

        maxX_maxY, minX_minY, max_x, min_x, max_y, min_y = findingMaxAndMinPoints(contours_list)
        maxX_minY, maxY_minX = finding_theMaxMinOrMinMaxPoints(contours_list, max_x, min_x, max_y, min_y)

        four_points = [minX_minY, maxX_minY, maxX_maxY, fourth_point]

        for i in range(len(four_points)):
            cv2.circle(original_image, (four_points[i][0], four_points[i][1]), 15, (0, 255, 0), -1)
            show_small_image("Black squares: ", original_image)

        print("First point: ", minX_minY)
        print("Second point: ", maxX_minY)
        print("Third point: ", maxX_maxY)
        print("Fourth point: ", fourth_point)


    return four_points


def finding_sectors(image):
    sectors = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 20, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C | cv2.THRESH_OTSU)[1]
    im2, cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.drawContours(image, cnts, -1, (0,255,0), 15)
    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            # cv2.drawContours(image, c, -1, (10, 200, 50), 50)
            # show_small_image(image)
            perimeter = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
            if len(approx) == 4 and cv2.contourArea(c) > 5000:
                sectors.append(approx)
                # cv2.drawContours(image, approx, -1, (0,255,0), 20)
                # show_small_image(image)
    sectors = contours.sort_contours(sectors, "top-to-bottom")[0]

    return sectors


def align_by_four_points(four_points, original_image):
    paper_cnts = np.array([four_points])
    paper = four_point_transform(original_image, paper_cnts.reshape(4, 2))

    return paper


def finding_the_bubbled_by_BlobDetector(sector_n):
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()
    #Filter by color
    params.filterByColor = True
    params.blobColor = 0

    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 140

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 1000

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.1

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3:
        pass
        # detector = cv2.SimpleBlobDetector(params)
    else:
        detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(sector_n)

    # print(keypoints)
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(sector_n, keypoints, np.array([]), (0, 255, 0),
                                            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow("Findind bubbled circles by Blob Detector: ", im_with_keypoints)
    cv2.waitKey(0)
    # show_small_image(sector_n)
    # print(im_with_keypoints)

    return im_with_keypoints, keypoints


def align_letters_with_contours(questionCnts, kz_alphabet, circle_area):

    array_aligned_with_contours_letters = []
    delta_y1 = 0
    delta_y2 = 2
    alphabet_counter = 0

    # print(questionCnts)
    for (q,i) in enumerate(np.arange(0, len(questionCnts), 18)):

        cnts = contours.sort_contours(questionCnts[i:i + 18])[0]
        for c in range(len(cnts)):
            # cv2.drawContours(circle_area, cnts[c], -1, (100, 230, 10), 10)
            # show_small_image(circle_area)
            # print(cnts[c][0][0][1])
            # print(delta_y1, delta_y2)
            if delta_y1 < cnts[c][0][0][1] <= delta_y2:
                # print("Letter is: ", kz_alphabet[alphabet_counter])
                # print("Between %f and %f" % (delta_y1, delta_y2))
                # print("Y of the contour is: ", cnts[c][0][0][1])
                # cv2.drawContours(circle_area, cnts[c], -1, (200, alphabet_counter, 100), 10)
                # show_small_image(circle_area)
                array_aligned_with_contours_letters.append([cnts[c], kz_alphabet[alphabet_counter]])
            if c == 17:
                delta_y1 = delta_y2
                delta_y2 += 50.5
                alphabet_counter += 1

    # print(ranked_contour_with_kz_alphabet_array)
    return array_aligned_with_contours_letters


def finding_the_circle_area(sector_n):

    gray = cv2.cvtColor(sector_n, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    ret,th1 = cv2.threshold(blurred,127,255,cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,11,2)
    th3 = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY,11,2)

    # cv2.imshow("Th1 sector: ", th1)
    # cv2.waitKey(0)
    # cv2.imshow("Th2 sector: ", th2)
    # cv2.waitKey(0)
    # cv2.imshow("Th3 sector: ", th3)
    # cv2.waitKey(0)
    
    cnts = cv2.findContours(th2.copy(), cv2.RETR_TREE,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    circle_area_cnts = []

    # cv2.imshow("Edged sector: ", edged)
    # cv2.waitKey(0)
    # ensure that at least one contour was found
    if len(cnts) > 0:
        # sort the contours according to their size in
        # descending order
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        # loop over the sorted contours
        for c in cnts:
            # cv2.drawContours(sector_n, c, -1, (0,255,0), 10)
            # show_small_image(sector_n)
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)

            # if our approximated contour has four points,
            # then we can assume we have found the paper
            if len(approx) == 4:
                circle_area_cnts.append(approx)
                cv2.drawContours(sector_n, approx, -1, (0,0,255), 15)
                show_small_image("Contours four point: ", sector_n)
                break

    circle_area = four_point_transform(sector_n, circle_area_cnts[0].reshape(4, 2))
    circle_area_warped = four_point_transform(gray, circle_area_cnts[0].reshape(4, 2))

    return circle_area, circle_area_warped


def finding_circles(circle_area, circle_area_warped):

    # show_small_image(circle_area)

    gray = cv2.cvtColor(circle_area, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edged = cv2.Canny(blurred, 110, 200)

    # cv2.imshow("Edged sector: ", edged)
    # cv2.waitKey(0)
    

    # show_small_image(circle_area_warped)
    
    # circle_area_warped = cv2.bitwise_not(circle_area_warped, circle_area_warped.copy())
    # show_small_image(circle_area_warped)

    ret,thresh = cv2.threshold(edged.copy(), 0, 255, cv2.THRESH_BINARY)
    # # Otsu's thresholding
    # ret2,th2 = cv2.threshold(edged,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # # Otsu's thresholding after Gaussian filtering
    # blur = cv2.GaussianBlur(edged,(5,5),0)
    # ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


    # cv2.imshow("Adaptive threshold sector: ", thresh)
    # cv2.waitKey(0)

   
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    questionCnts = []

    bubbled = []
    
    # loop over the contours
    for c in cnts:
        # print(c)
        # compute the bounding box of the contour, then use the
        # bounding box to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # in order to label the contour as a question, region
        # should be sufficiently wide, sufficiently tall, and
        # have an aspect ratio approximately equal to 1

        if w >= 20 and h >= 20 and ar >= 0.8 and ar <= 1.1:
            questionCnts.append(c)
            # cv2.drawContours(circle_area)

    questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
    questionCnts = contours.sort_contours(questionCnts, method="left-to-right")[0]
    
    # cv2.drawContours(circle_area, questionCnts, -1, (0,255,0), 10)
    # show_small_image(circle_area)
    
    return questionCnts, thresh, circle_area


def identification_of_characters_with_coordinates(kz_alphabet):
    range_of_character = []

    delta_y1 = 0
    delta_y2 = 0
    alphabet_counter = 0

    for l in kz_alphabet:
        delta_y2 += 50
        range_of_character.append([delta_y1, l, delta_y2])
        delta_y1 = delta_y2
    # print(range_of_character)
    return range_of_character


def identification_of_numbers_s3(numbers_for_s3):
    range_of_numbers = []
    delta_y1 = 150
    delta_y2 = 150
    alphabet_counter = 0

    for l in numbers_for_s3:
        delta_y2 += 48
        range_of_numbers.append([delta_y1, l, delta_y2])
        delta_y1 = delta_y2
    print(range_of_numbers)
    return range_of_numbers


def identification_of_numbers_s4(numbers_for_s4):
    range_of_numbers = []
    delta_y1 = 0
    delta_y2 = 50
    delta_x1 = 0
    delta_x2 = 0

    alphabet_counter = 0
    counter = 0
    for l in numbers_for_s4:
        counter += 1
        if counter == 7:
            delta_y1 = delta_y2
            delta_y2 += 50
            counter = 0
        delta_x2 += 50.5
        range_of_numbers.append([delta_x1, delta_y1, l, delta_x2, delta_y2])
        delta_x1 = delta_x2

    return range_of_numbers


def sorting_keypoints_by_X(key_points):
    x_y_of_keypoints = []

    for i in range(len(key_points)):
        x_y_of_keypoints.append([key_points[i].pt[0], key_points[i].pt[1]])
    sorted_xy_keypoints = sorted(x_y_of_keypoints)

    # print("Sorted by X keypoints: %s" % (sorted_xy_keypoints))

    return sorted_xy_keypoints


def finding_the_bubbled(sorted_xy_keypoints, range_of_character):
    bubbled_characters = []
    bubbled_str = ""
    for s in range(len(sorted_xy_keypoints)):
        for r in range(len(range_of_character)):
            if range_of_character[r][0] < sorted_xy_keypoints[s][1] < range_of_character[r][2]:
                bubbled_str += str(range_of_character[r][1])
                bubbled_characters.append(range_of_character[r][1])
    bubbled_characters.append(bubbled_str)
    # print(bubbled_characters)
    return bubbled_characters


def finding_the_bubbled_for_s4(sorted_xy_keypoints, range_of_character):
    bubbled_characters = []
    bubbled_str = ""
    # delta_x1, delta_y1, l, delta_x2, delta_y2
    for s in range(len(sorted_xy_keypoints)):
        for r in range(len(range_of_character)):
            if range_of_character[r][0] < sorted_xy_keypoints[s][0] < range_of_character[r][3] and range_of_character[r][1] < sorted_xy_keypoints[s][1] < range_of_character[r][4]:
                bubbled_str += range_of_character[r][2]
                bubbled_characters.append(range_of_character[r][2])
    bubbled_characters.append(bubbled_str)
    # print(bubbled_characters)
    return bubbled_characters


# def find_circles(warped_image):
#     img = cv2.medianBlur(warped_image, 5)
#     cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#
#     circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
#                                param1=50, param2=30, minRadius=0, maxRadius=0)
#
#     circles = np.uint16(np.around(circles))
#
#     for i in circles[0, :]:
#         # draw the outer circle
#
#         cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
#         # draw the center of the circle
#         cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
#
#     return cimg
#
# def sort_the_circles(circles):
#     for i in range(len(circles[0])):
#         if i == len(circles):
#             break
#         if circles[0][i][0] > circles[0][i + 1][0]:
#             temp = circles[0][i + 1]
#             circles[0][i + 1] = circles[0][i]
#             circles[0][i] = temp
#     return circles


def sort_the_circles_ByX(circles):
    circles = sorted(circles[0], key=lambda row: row[0])
    return circles


def sort_the_circles_ByY(circles):
    circles = sorted(circles[0], key=lambda row: row[1])
    return circles


def sort_per_column(circles, numberOfRows):

    per_column = []
    print("Rows:  Range:")
    for (row, i) in enumerate(np.arange(0, len(circles), numberOfRows)):
        print("Number of rows: ", numberOfRows)
        per_column.append(circles[i:i+numberOfRows])
        print(row, i)

    for i in range(len(per_column)):
        # for j in i:
        sorted_per_column = sorted(per_column[i], key=lambda row: row[1])
        per_column[i] = sorted_per_column

    return per_column


def sort_per_row(circles, numberOfColumns):

    per_row = []
    print("Cols:  Range:")
    for (row, i) in enumerate(np.arange(0, len(circles), numberOfColumns)):
        print("Number of columns: ", numberOfColumns)
        per_row.append(circles[i:i+numberOfColumns])
        print(row, i)

    for i in range(len(per_row)):
        sorted_per_column = sorted(per_row[i], key=lambda row: row[0])
        per_row[i] = sorted_per_column

    return per_row


def finding_the_circles_by_Hough_Circles(img):

    img = cv2.medianBlur(img, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # edged = cv2.Canny(img, 100, 200)
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                            cv2.THRESH_BINARY, 11, 2)
    # Otsu's thresholding
    ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imshow("Thresholding 'Hough Circles': ", th3)
    cv2.imshow("Thresholding 'Hough Circles' 2 :", th2)
    cv2.waitKey(0)

    circles = cv2.HoughCircles(th3.copy(), cv2.HOUGH_GRADIENT, 1, 45, param1=25, param2=17, minRadius=8, maxRadius=22)

    circles = np.uint16(np.around(circles))
    # print(circles)
    # circles = sorted(circles[0], key=lambda row: row[0])

    print("Length of circles: ", len(circles))

    for i in circles[0, :]:
        cv2.circle(cimg, (i[0], i[1]), i[2], (128, 244, 66), 5)
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow("Circles: ", cimg)
    cv2.waitKey(0)
    show_small_image("Circles: ", cimg)
    cv2.destroyAllWindows()

    return circles


def show_keypoints(keypoints):

    print("Keypoints: ", keypoints)
    print("Length of keypoints: ", len(keypoints))


def align_circles_with_letters(sorted_c_a, parameter1):
    print("Sorted Circles: ", sorted_c_a)
    alignedCircWithLettList = []
    circles_not_related = []
    counter = 0
    d_x_minus = 0
    d_x_plus = 0
    d_y_minus = 0
    d_y_plus = 0
    for i in range(len(sorted_c_a)):
        print("Length of the sorted circle area of one column", len(sorted_c_a[i]))
        for j in range(len(sorted_c_a[i])):
            x = sorted_c_a[i][j][0]
            y = sorted_c_a[i][j][1]
            r = sorted_c_a[i][j][2]
            d_x_minus = x - r
            d_x_plus = x + r
            d_y_minus = y - r
            d_y_plus = y + r
            alignedCircWithLettList.append([d_x_minus, d_x_plus, d_y_minus, d_y_plus, str(parameter1[j])])
    print("Aligned Circles With Letters List: ", alignedCircWithLettList)
    return alignedCircWithLettList


def finding_the_bubbled_characters(sorted_xy_keypoints, alignedCircWithLettList):
    characters = []
    combined_characters = ''
    for j in range(len(sorted_xy_keypoints)):
        for i in range(len(alignedCircWithLettList)):
            if alignedCircWithLettList[i][0] < sorted_xy_keypoints[j][0] < alignedCircWithLettList[i][1] and alignedCircWithLettList[i][2] < sorted_xy_keypoints[j][1] < alignedCircWithLettList[i][3]:
                    combined_characters += str(alignedCircWithLettList[i][4])
                    characters.append(alignedCircWithLettList[i][4])
    characters.append(combined_characters)
    return characters


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


def findingMaxAndMinPoints(contours_list):

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

    return maxX_maxY, minX_minY, max_x, min_x, max_y, min_y


def finding_theMaxMinOrMinMaxPoints(contours_list, max_x, min_x, max_y, min_y):
    maxX_minY= None
    maxY_minX = None

    for i in range(len(contours_list)):
        if min_y <= contours_list[i][1] <= min_y + 30 and max_x - 30 <= contours_list[i][0] <= max_x:
            maxX_minY = contours_list[i]
            # print(contours_list[i])
            # cv2.circle(original_image, (contours_list[i][0], contours_list[i][1]), 15, (0, 255, 0),-1)
            #     # cv2.drawContours(original_image, black_squares[i][j], -1, (0, 255, 0), 15)
            # show_small_image("Black squares: ", original_image)
        if min_x <= contours_list[i][0] <= min_x + 30 and max_y - 30 <= contours_list[i][1] <= max_y:
            print(contours_list[i])
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

    return fourth_point


def main():
    original_image = cv2.imread('images/not_empty/Scan_0023.jpg')


    kz_alphabet = np.array(['А', 'Ә', 'Б', 'В', 'Г', 'Ғ', 'Д', 'Е', 'Ж', 'З', 'И', 'Й',
                            'К', 'Қ', 'Л', 'М', 'Н', 'Ң', 'О', 'Ө', 'П', 'Р', 'С', 'Т',
                            'У', 'Ұ', 'Ү', 'Ф', 'Х', 'Һ', 'Ц', 'Ч', 'Ш', 'Щ', 'Ь', 'Ы',
                            'І', 'Э', 'Ю', 'Я', '-'])


    numbers_for_s3_s4 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])

    kz_alphabet_for_s5 = np.array(['А', 'Ә', 'Б', 'В', 'Г', 'Ғ',
                                   'Д', 'Е', 'Ж', 'З', 'И', 'К',
                                   'Л', 'М', 'Н', 'О', 'П', 'Р',
                                   'С', 'У', 'Ф', 'Х', 'Э', 'Ю'])


    numbers_for_s6 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])

    black_squares = finding_the_black_squares(original_image)

    print(len(black_squares))
    four_points = check_for_four_points(black_squares, original_image)


    for i in range(len(black_squares)):
        cv2.drawContours(original_image, black_squares[i], -1, (240, 200, 100), 15)


    paper = align_by_four_points(four_points, original_image)
    show_small_image("Paper: ", paper)

    sectors = finding_sectors(paper)
    sectors = sorted(sectors, key=cv2.contourArea, reverse=True)

    # Finding sectors
    sector1 = paper[2345:3360, 0:2185]
    sector2 = paper[1305:2355, 0:2185]
    sector3 = paper[565:1295, 600:1140]
    sector4 = paper[175:485, 600:1140]
    sector5 = paper[925:1315, 1300:1520]
    sector6 = paper[145:515, 1300:1520]
    show_small_image("Sector: ", sector1)
    # sector2 = four_point_transform(paper, sectors[0].reshape(4, 2))
    # sector3 = four_point_transform(paper, sectors[2].reshape(4, 2))
    # sector4 = four_point_transform(paper, sectors[5].reshape(4, 2))
    # sector5 = four_point_transform(paper, sectors[7].reshape(4, 2))
    # sector6 = four_point_transform(paper, sectors[6].reshape(4, 2))
    # sector7 = four_point_transform(paper, sectors[4].reshape(4, 2))
    #
    # Kerekti zherin gana alamiz
    # sector3 = sector3[0:1000, 175:1000]
    # sector4 = sector4[0:1000, 175:1000]

    # 90 gradusqa ainaldiramiz
    rotated_sector1 = imutils.rotate_bound(sector1, 90)
    rotated_sector2 = imutils.rotate_bound(sector2, 90)
    rotated_sector3 = imutils.rotate_bound(sector3, 90)
    rotated_sector4 = imutils.rotate_bound(sector4, 90)
    rotated_sector5 = imutils.rotate_bound(sector5, 90)
    rotated_sector6 = imutils.rotate_bound(sector6, 90)

    show_small_image("Sector 5: ", rotated_sector6)
    # # rotated_sector7 = imutils.rotate_bound(sector7, 90)
    #
    # # Finding circle area
    circle_area1, circle_area1_warped  = finding_the_circle_area(rotated_sector1)
    circle_area2, circle_area2_warped  = finding_the_circle_area(rotated_sector2)
    circle_area3, circle_area3_warped = finding_the_circle_area(rotated_sector3)
    circle_area4, circle_area4_warped = finding_the_circle_area(rotated_sector4)
    circle_area5, circle_area5_warped = finding_the_circle_area(rotated_sector5)
    circle_area6, circle_area6_warped = finding_the_circle_area(rotated_sector6)
    #
    # # Append all circle areas to the list
    list_of_circle_area = []
    list_of_circle_area.append(circle_area1)
    list_of_circle_area.append(circle_area2)
    list_of_circle_area.append(circle_area3)
    list_of_circle_area.append(circle_area4)
    list_of_circle_area.append(circle_area5)
    list_of_circle_area.append(circle_area6)

    # Convert all .jpg files into .png files and save. Hough Circles function work with .png files
    c = 1
    for i in range(len(list_of_circle_area)):
        cv2.imwrite('images/png_format_circle_area/circle_area%d.png'%(c), list_of_circle_area[i])
        c += 1
        if i == len(list_of_circle_area) - 1:
            break

    # Reading the .png files
    circle_area1 = cv2.imread("images/png_format_circle_area/circle_area1.png", 0)
    circle_area2 = cv2.imread("images/png_format_circle_area/circle_area2.png", 0)
    circle_area3 = cv2.imread("images/png_format_circle_area/circle_area3.png", 0)
    circle_area4 = cv2.imread("images/png_format_circle_area/circle_area4.png", 0)
    circle_area5 = cv2.imread("images/png_format_circle_area/circle_area5.png", 0)
    circle_area6 = cv2.imread("images/png_format_circle_area/circle_area6.png", 0)

    # Finding the circles
    circles_s1 = finding_the_circles_by_Hough_Circles(circle_area1)
    circles_s2 = finding_the_circles_by_Hough_Circles(circle_area2)
    circles_s3 = finding_the_circles_by_Hough_Circles(circle_area3)
    circles_s4 = finding_the_circles_by_Hough_Circles(circle_area4)
    circles_s5 = finding_the_circles_by_Hough_Circles(circle_area5)
    circles_s6 = finding_the_circles_by_Hough_Circles(circle_area6)

    # show_small_image("Circles_sn: ", circles_s5)

    # Sorting the circles by X or by Y
    circles_s1 = sort_the_circles_ByX(circles_s1)
    circles_s2 = sort_the_circles_ByX(circles_s2)
    circles_s3 = sort_the_circles_ByX(circles_s3)
    circles_s4 = sort_the_circles_ByX(circles_s4)
    circles_s5 = sort_the_circles_ByY(circles_s5)
    circles_s6 = sort_the_circles_ByY(circles_s6)

    # Sorting per row or per column
    sorted_c_a1 = sort_per_column(circles_s1, numberOfRows=41)
    sorted_c_a2 = sort_per_column(circles_s2, numberOfRows=41)
    sorted_c_a3 = sort_per_column(circles_s3, numberOfRows=10)
    sorted_c_a4 = sort_per_column(circles_s4, numberOfRows=10)
    sorted_c_a5 = sort_per_row(circles_s5, numberOfColumns=6)
    sorted_c_a6 = sort_per_row(circles_s6, numberOfColumns=6)

    # Align circles with letters
    alignedCircWithLettList_s1 = align_circles_with_letters(sorted_c_a1, kz_alphabet)
    alignedCircWithLettList_s2 = align_circles_with_letters(sorted_c_a2, kz_alphabet)
    alignedCircWithNumbList_s3 = align_circles_with_letters(sorted_c_a3, numbers_for_s3_s4)
    alignedCircWithNumbList_s4 = align_circles_with_letters(sorted_c_a4, numbers_for_s3_s4)
    alignedCircWithLettList_s5 = align_circles_with_letters(sorted_c_a5, kz_alphabet_for_s5)
    alignedCircWithLettList_s6 = align_circles_with_letters(sorted_c_a6, numbers_for_s6)

    # Finding the bubbled keypoints by Blob Detector function
    im_with_keypoints1, keypoints1 = finding_the_bubbled_by_BlobDetector(circle_area1)
    im_with_keypoints2, keypoints2 = finding_the_bubbled_by_BlobDetector(circle_area2)
    im_with_keypoints3, keypoints3 = finding_the_bubbled_by_BlobDetector(circle_area3)
    im_with_keypoints4, keypoints4 = finding_the_bubbled_by_BlobDetector(circle_area4)
    im_with_keypoints5, keypoints5 = finding_the_bubbled_by_BlobDetector(circle_area5)
    im_with_keypoints6, keypoints6 = finding_the_bubbled_by_BlobDetector(circle_area6)
    # # im_with_keypoints7, keypoints7 = finding_the_bubbled_by_BlobDetector(circle_area7)

    # Sorting keypoints by X
    sorted_xy_keypoints_s1 = sorting_keypoints_by_X(keypoints1)
    sorted_xy_keypoints_s2 = sorting_keypoints_by_X(keypoints2)
    sorted_xy_keypoints_s3 = sorting_keypoints_by_X(keypoints3)
    sorted_xy_keypoints_s4 = sorting_keypoints_by_X(keypoints4)
    sorted_xy_keypoints_s5 = sorting_keypoints_by_X(keypoints5)
    sorted_xy_keypoints_s6 = sorting_keypoints_by_X(keypoints6)

    # Print keypoints
    show_keypoints(sorted_xy_keypoints_s1)
    show_keypoints(sorted_xy_keypoints_s2)
    show_keypoints(sorted_xy_keypoints_s3)
    show_keypoints(sorted_xy_keypoints_s4)
    show_keypoints(sorted_xy_keypoints_s5)
    show_keypoints(sorted_xy_keypoints_s6)

    characters_s1 = finding_the_bubbled_characters(sorted_xy_keypoints_s1, alignedCircWithLettList_s1)
    characters_s2 = finding_the_bubbled_characters(sorted_xy_keypoints_s2, alignedCircWithLettList_s2)
    characters_s3 = finding_the_bubbled_characters(sorted_xy_keypoints_s3, alignedCircWithNumbList_s3)
    characters_s4 = finding_the_bubbled_characters(sorted_xy_keypoints_s4, alignedCircWithNumbList_s4)
    characters_s5 = finding_the_bubbled_characters(sorted_xy_keypoints_s5, alignedCircWithLettList_s5)
    characters_s6 = finding_the_bubbled_characters(sorted_xy_keypoints_s6, alignedCircWithLettList_s6)

    print(characters_s1)
    print(characters_s2)
    print(characters_s3)
    print(characters_s4)
    print(characters_s5)
    print(characters_s6)

    workbook = xlsxwriter.Workbook('applicants.xlsx')
    worksheet = workbook.add_worksheet()


    # Add a bold format to use to highlight cells.
    bold = workbook.add_format({'bold': True})


    # Add a number format for cells with money.
    money = workbook.add_format({'num_format': '$#,##0'})


    # Write some data headers.
    worksheet.write('A1', 'Тегі-Фамилия', bold)
    worksheet.write('B1', 'Аты-Имя', bold)
    worksheet.write('C1', 'ЖСН-ИИН', bold)
    worksheet.write('D1', 'Нұсқа-Вариант', bold)
    worksheet.write('E1', 'Сынып литерасы-Литера класса', bold)
    worksheet.write('F1', 'Қосымша сектор-Резервный сектор', bold)


    row = 1
    col = 0
    worksheet.write(row, col, characters_s1[-1])
    worksheet.write(row, col + 1, characters_s2[-1])
    worksheet.write(row, col + 2, characters_s3[-1])
    worksheet.write(row, col + 3, characters_s4[-1])
    worksheet.write(row, col + 4, characters_s5[-1])
    worksheet.write(row, col + 5, characters_s6[-1])

    workbook.close()

    # Another method
    # bubbled_characters_s1 = finding_the_bubbled(sorted_xy_keypoints_s1, range_of_character)
    # bubbled_characters_s2 = finding_the_bubbled(sorted_xy_keypoints_s2, range_of_character)
    # bubbled_characters_s3 = finding_the_bubbled(sorted_xy_keypoints_s3, range_of_numbers_s3)
    # bubbled_characters_s4 = finding_the_bubbled(sorted_xy_keypoints_s4, range_of_numbers_s4)
    # bubbled_characters_s5 = finding_the_bubbled_for_s4(sorted_xy_keypoints_s5, range_of_numbers_s5)


if __name__ == "__main__":
    main()