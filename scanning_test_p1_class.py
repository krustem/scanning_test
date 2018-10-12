#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
from imutils import contours
from imutils.perspective import four_point_transform

import cv2
import numpy as np
import imutils
import xlsxwriter

from PyQt5.QtCore import *
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore
from PyQt5.QtGui import QImage

class EnginePart:
    def __init__(self):
        super().__init__()


    def finding_the_bubbled_characters(self, sorted_xy_keypoints, alignedCircWithLettList):
        characters = []
        combined_characters = ''
        for j in range(len(sorted_xy_keypoints)):
            for i in range(len(alignedCircWithLettList)):
                if alignedCircWithLettList[i][0] < sorted_xy_keypoints[j][0] < alignedCircWithLettList[i][1] and \
                        alignedCircWithLettList[i][2] < sorted_xy_keypoints[j][1] < alignedCircWithLettList[i][3]:
                    combined_characters += alignedCircWithLettList[i][4]
                    characters.append(alignedCircWithLettList[i][4])
        characters.append(combined_characters)
        return characters


    def show_keypoints(self, keypoints):
        print("Keypoints: ", keypoints)
        print("Length of keypoints: ", len(keypoints))


    def sorting_keypoints_by_X(self, key_points):
        x_y_of_keypoints = []

        for i in range(len(key_points)):
            x_y_of_keypoints.append([key_points[i].pt[0], key_points[i].pt[1]])
        sorted_xy_keypoints = sorted(x_y_of_keypoints)

        return sorted_xy_keypoints


    def finding_the_bubbled_by_BlobDetector(self, sector_n):
        params = cv2.SimpleBlobDetector_Params()
        # Filter by color
        params.filterByColor = True
        params.blobColor = 0

        # Change thresholds
        params.minThreshold = 0
        params.maxThreshold = 140

        # Filter by Area.
        params.filterByArea = True
        params.minArea = 600

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
        else:
            detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs.
        keypoints = detector.detect(sector_n)

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(sector_n, keypoints, np.array([]), (0, 255, 0),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        return im_with_keypoints, keypoints


    def finding_the_circles_by_Hough_Circles(self, img):
        img = cv2.medianBlur(img, 5)
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                    cv2.THRESH_BINARY, 11, 2)
        # Otsu's thresholding
        ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        circles = cv2.HoughCircles(th3, cv2.HOUGH_GRADIENT, 1, 40, param1=50, param2=17, minRadius=10, maxRadius=22)

        circles = np.uint16(np.around(circles))

        print("Length of circles: ", len(circles[0, :]))

        for i in circles[0, :]:
            cv2.circle(cimg, (i[0], i[1]), i[2], (128, 244, 66), 5)
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

        return circles


    def sort_the_circles_ByX(self, circles):
        circles = sorted(circles[0], key=lambda row: row[0])
        return circles


    def sort_the_circles_ByY(self, circles):
        circles = sorted(circles[0], key=lambda row: row[1])
        return circles


    def sort_per_column(self, circles, numberOfRows):
        per_column = []
        print("Rows:  Range:")
        for (row, i) in enumerate(np.arange(0, len(circles), numberOfRows)):
            print("Number of rows: ", numberOfRows)
            per_column.append(circles[i:i + numberOfRows])
            print(row, i)

        for i in range(len(per_column)):
            # for j in i:
            sorted_per_column = sorted(per_column[i], key=lambda row: row[1])
            per_column[i] = sorted_per_column

        return per_column


    def sort_per_row(self, circles, numberOfColumns):
        per_row = []
        print("Cols:  Range:")
        for (row, i) in enumerate(np.arange(0, len(circles), numberOfColumns)):
            print("Number of columns: ", numberOfColumns)
            per_row.append(circles[i:i + numberOfColumns])
            print(row, i)

        for i in range(len(per_row)):
            sorted_per_column = sorted(per_row[i], key=lambda row: row[0])
            per_row[i] = sorted_per_column

        return per_row


    def align_circles_with_letters(self, sorted_c_a, parameter1):
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


    def finding_the_circle_area(self, sector_n):
        gray = cv2.cvtColor(sector_n, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)

        ret, th1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        th2 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                    cv2.THRESH_BINARY, 11, 2)
        th3 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)

        cnts = cv2.findContours(th2.copy(), cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        circle_area_cnts = []

        # ensure that at least one contour was found
        if len(cnts) > 0:
            # sort the contours according to their size in
            # descending order
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

            # loop over the sorted contours
            for c in cnts:
                # approximate the contour
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)

                # if our approximated contour has four points,
                # then we can assume we have found the paper
                if len(approx) == 4:
                    circle_area_cnts.append(approx)
                    break

        circle_area = four_point_transform(sector_n, circle_area_cnts[0].reshape(4, 2))
        circle_area_warped = four_point_transform(gray, circle_area_cnts[0].reshape(4, 2))

        return circle_area, circle_area_warped


    def finding_sectors(self, paper):
        sectors = []
        gray = cv2.cvtColor(paper, cv2.COLOR_BGR2GRAY)
        # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(gray, 20, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C | cv2.THRESH_OTSU)[1]
        im2, cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # cv2.drawContours(image, cnts, -1, (0,255,0), 15)
        if len(cnts) > 0:
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
            for c in cnts:
                perimeter = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
                if len(approx) == 4 and cv2.contourArea(c) > 5000:
                    sectors.append(approx)
        sectors = contours.sort_contours(sectors, "top-to-bottom")[0]

        return sectors


    def align_by_four_points(self, four_points, original_image):
        paper_cnts = np.array([four_points])
        paper = four_point_transform(original_image, paper_cnts.reshape(4, 2))

        return paper


    def check_for_four_points(self, black_squares, original_image):
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
            maxX_maxY, minX_minY, max_x, min_x, max_y, min_y = self.findingMaxAndMinPoints(contours_list)
            maxX_minY, maxY_minX = self.finding_theMaxMinOrMinMaxPoints(contours_list, max_x, min_x, max_y, min_y)

            four_points = [minX_minY, maxX_minY, maxX_maxY, maxY_minX]

            print(four_points)

            for i in range(len(four_points)):
                cv2.circle(original_image, (four_points[i][0], four_points[i][1]), 15, (0, 255, 0), -1)
                # show_small_image("Black squares: ", original_image)

            print("First point: ", minX_minY)
            print("Second point: ", maxX_minY)
            print("Third point: ", maxX_maxY)
            print("Fourth point: ", maxY_minX)

        else:

            fourth_point = self.addManuallyOnePoint(original_image)

            contours_list = []
            for i in range(len(black_squares)):
                # print(black_squares[i][0])
                for j in range(len(black_squares[i][0])):
                    contours_list.append([black_squares[i][0][j][0][0], black_squares[i][0][j][0][1]])
            print("Contours List: ", contours_list)

            sorted_contours_list_by_X = sorted(contours_list, key=lambda row: row[0])
            contours_list = sorted_contours_list_by_X

            maxX_maxY, minX_minY, max_x, min_x, max_y, min_y = self.findingMaxAndMinPoints(contours_list)
            maxX_minY, maxY_minX = self.finding_theMaxMinOrMinMaxPoints(contours_list, max_x, min_x, max_y, min_y)

            four_points = [minX_minY, maxX_minY, maxX_maxY, fourth_point]

            for i in range(len(four_points)):
                cv2.circle(original_image, (four_points[i][0], four_points[i][1]), 15, (0, 255, 0), -1)
                # show_small_image("Black squares: ", original_image)

            print("First point: ", minX_minY)
            print("Second point: ", maxX_minY)
            print("Third point: ", maxX_maxY)
            print("Fourth point: ", fourth_point)

        return four_points


    def displayImage(self, image, imgLabel):
        qformat = QImage.Format_Indexed8
        if len(image.shape)==3:
            if (image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888
        img=QImage(image, image.shape[1], image.shape[0], image.strides[0], qformat)
        img = img.rgbSwapped()
        pixmap = QPixmap.fromImage(img)
        smaller_pixmap = pixmap.scaled(600, 750, Qt.KeepAspectRatio, Qt.FastTransformation)
        imgLabel.setPixmap(smaller_pixmap)
        imgLabel.setAlignment(QtCore.Qt.AlignHCenter|QtCore.Qt.AlignVCenter)


    def addManuallyOnePoint(self, original_image):
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
                            # show_small_image("Black squares: ", original_image)

        return fourth_point


    def finding_theMaxMinOrMinMaxPoints(self, contours_list, max_x, min_x, max_y, min_y):
        maxX_minY = None
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


    def takeXPoints(self, contours_list):
        x_points = []
        for i in range(len(contours_list)):
            x_point = contours_list[i][0]
            x_points.append(x_point)
        return x_points


    def takeYPoints(self, contours_list):
        y_points = []
        for i in range(len(contours_list)):
            y_point = contours_list[i][1]
            y_points.append(y_point)
        return y_points


    def findingMaxAndMinPoints(self, contours_list):

        x_points = self.takeXPoints(contours_list)
        y_points = self.takeYPoints(contours_list)

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


    def finding_the_black_squares(self, image):

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
                    cv2.drawContours(image, approx, -1, (0, 0, 255), 15)
                    # show_small_image("Black squares: ", image)
        # print(black_squares)
        return black_squares