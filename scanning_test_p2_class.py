#!/usr/bin/env python
# -*- coding: utf-8 -*-


import cv2
import numpy as np
import imutils
import xlsxwriter
from imutils import contours
from imutils.perspective import four_point_transform



class EnginePart2:
    def __init__(self):
        super().__init__()

    # def show_small_image(self, image_description, image):
    #     small = cv2.resize(image, (0, 0), fx=0.29, fy=0.29)
    #     cv2.imshow(image_description, small)
    #     cv2.waitKey(0)



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
                    cv2.drawContours(image, approx, -1, (0,0,255), 15)
                    # show_small_image("Black squares: ", image)
        # print(black_squares)
        return black_squares

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

            # for i in range(len(four_points)):
            #     cv2.circle(original_image, (four_points[i][0], four_points[i][1]), 15, (0, 255, 0), -1)
            #     show_small_image("Black squares: ", original_image)

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

            # for i in range(len(four_points)):
            #     cv2.circle(original_image, (four_points[i][0], four_points[i][1]), 15, (0, 255, 0), -1)
            #     show_small_image("Black squares: ", original_image)

            print("First point: ", minX_minY)
            print("Second point: ", maxX_minY)
            print("Third point: ", maxX_maxY)
            print("Fourth point: ", fourth_point)


        return four_points


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



    def finding_theMaxMinOrMinMaxPoints(self, contours_list, max_x, min_x, max_y, min_y):
        maxX_minY= None
        maxY_minX = None

        for i in range(len(contours_list)):
            if min_y <= contours_list[i][1] <= min_y + 60 and max_x - 60 <= contours_list[i][0] <= max_x:
                maxX_minY = contours_list[i]
                # print(contours_list[i])
                # cv2.circle(original_image, (contours_list[i][0], contours_list[i][1]), 15, (0, 255, 0),-1)
                #     # cv2.drawContours(original_image, black_squares[i][j], -1, (0, 255, 0), 15)
                # show_small_image("Black squares: ", original_image)
            if min_x <= contours_list[i][0] <= min_x + 60 and max_y - 60 <= contours_list[i][1] <= max_y:
                print(contours_list[i])
                maxY_minX = contours_list[i]
                # print(contours_list[i])
                # cv2.circle(original_image, (contours_list[i][0], contours_list[i][1]), 15, (0, 255, 0), -1)
                #     # cv2.drawContours(original_image, black_squares[i][j], -1, (0, 255, 0), 15)
                # show_small_image("Black squares: ", original_image)
        return maxX_minY, maxY_minX


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


    def finding_sectors(self, image):
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


    def finding_the_circle_area(self, sector_n):
        gray = cv2.cvtColor(sector_n, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)

        ret, th1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
        th2 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                    cv2.THRESH_BINARY, 11, 2)
        th3 = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)

        # cv2.imshow("Th1 sector: ", th1)
        # cv2.waitKey(0)
        # cv2.imshow("Th2 sector: ", th2)
        # cv2.waitKey(0)
        # cv2.imshow("Th3 sector: ", th3)
        # cv2.waitKey(0)

        cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
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
                    cv2.drawContours(sector_n, approx, -1, (0, 0, 255), 15)
                    # show_small_image("Contours four point: ", sector_n)
                    break

        circle_area = four_point_transform(sector_n, circle_area_cnts[0].reshape(4, 2))
        circle_area_warped = four_point_transform(gray, circle_area_cnts[0].reshape(4, 2))

        return circle_area, circle_area_warped


    def finding_the_circles_by_Hough_Circles(self, img):

        img = cv2.medianBlur(img, 5)
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                cv2.THRESH_BINARY, 11, 2)
        # Otsu's thresholding
        ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # cv2.imshow("Thresholding 'Hough Circles': ", th3)
        # cv2.imshow("Thresholding 'Hough Circles' 2 :", th2)
        # cv2.waitKey(0)

        circles = cv2.HoughCircles(th3, cv2.HOUGH_GRADIENT, 1, 40, param1=25, param2=17, minRadius=8, maxRadius=22)

        circles = np.uint16(np.around(circles))
        # print(circles)
        # circles = sorted(circles[0], key=lambda row: row[0])

        print("Length of circles: ", len(circles[0]))

        for i in circles[0, :]:
            cv2.circle(cimg, (i[0], i[1]), i[2], (128, 244, 66), 5)
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

        # cv2.imshow("Circles: ", cimg)
        # cv2.waitKey(0)
        # show_small_image("Circles: ", cimg)
        # cv2.destroyAllWindows()

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
            per_column.append(circles[i:i+numberOfRows])
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
            per_row.append(circles[i:i+numberOfColumns])
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


    def finding_the_bubbled_by_BlobDetector(self, sector_n):
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

        # cv2.imshow("Findind bubbled circles by Blob Detector: ", im_with_keypoints)
        # cv2.waitKey(0)
        # show_small_image(sector_n)
        # print(im_with_keypoints)

        return im_with_keypoints, keypoints



    def sorting_keypoints_by_Y(self, key_points):
        x_y_of_keypoints = []
        print(type(key_points))
        for i in range(len(key_points)):
            x_y_of_keypoints.append([key_points[i].pt[0], key_points[i].pt[1]])
        # sorted_xy_keypoints = sorted(x_y_of_keypoints)
        sorted_xy_keypoints = sorted(x_y_of_keypoints, key=lambda row: row[1])

        print("Sorted by Y keypoints: %s" % (sorted_xy_keypoints))

        return sorted_xy_keypoints


    def sorting_keypoints_by_X(self, key_points):
        print(type(key_points))
        if type(key_points) == list:
            sorted_xy_keypoints = sorted(key_points, key=lambda row: row[0])
        else:
            x_y_of_keypoints = []

            for i in range(len(key_points)):
                x_y_of_keypoints.append([key_points[i].pt[0], key_points[i].pt[1]])
            # sorted_xy_keypoints = sorted(x_y_of_keypoints)
            sorted_xy_keypoints = sorted(x_y_of_keypoints, key=lambda row: row[0])

            # print("Sorted by X keypoints: %s" % (sorted_xy_keypoints))

        return sorted_xy_keypoints


    def show_keypoints(self, keypoints):

        print("Keypoints: ", keypoints)
        print("Length of keypoints: ", len(keypoints))


    def finding_the_bubbled_characters(self, sorted_xy_keypoints, alignedCircWithLettList):
        characters = []
        combined_characters = ''
        for j in range(len(sorted_xy_keypoints)):
            for i in range(len(alignedCircWithLettList)):
                if alignedCircWithLettList[i][0] < sorted_xy_keypoints[j][0] < alignedCircWithLettList[i][1] and alignedCircWithLettList[i][2] < sorted_xy_keypoints[j][1] < alignedCircWithLettList[i][3]:
                        combined_characters += str(alignedCircWithLettList[i][4])
                        characters.append(alignedCircWithLettList[i][4])
        characters.append(combined_characters)
        return characters



# def main():
#     print("Second paper is started!")
#     original_image = cv2.imread('images/not_empty/Scan_0012.jpg')
#
#     LETTERS = np.array(['A','B','C','D','E'])
#     LETTERS2 = np.array(['A','B','C','D','E','F','G','H'])
#     SUBJECTS1 = np.array(['Қазақ тілі және әдебиет', 'Орыс тілі және әдебиет','Адам. Қоғам. Құқық.', 'Математика'])
#     SUBJECTS2 = np.array(['Химия', 'Физика', 'Биология', 'Тарих'])
#     SUBJECTS3 = np.array(['География', 'Ағылшын тілі', 'Неміс тілі', 'Француз тілі'])
#     ANSWER_KEY = {0:'B', 1:'C',2:'D',3:'E',4:'E',5:'C',6:'D',7:'C',8:'A',9:'A',10:'D',11:'B',12:'B', 13:'D', 14:'A', 15:'C', 16:'D', 17:'E', 18:'A',19:'D'}
#
#
#     black_squares = finding_the_black_squares(original_image)
#
#     print(len(black_squares))
#     #
#     # for i in range(len(black_squares)):
#     #     cv2.drawContours(original2_image, black_squares[i], -1, (200, 240, 100), 15)
#
#     four_points = check_for_four_points(black_squares, original_image)
#
#     for i in range(len(black_squares)):
#         cv2.drawContours(original_image, black_squares[i], -1, (200, 240, 100), 15)
#
#     paper = align_by_four_points(four_points, original_image)
#
#     sectors = finding_sectors(paper)
#     sectors = sorted(sectors, key=cv2.contourArea, reverse=True)
#
#     # Finding sectors
#     sector8 = paper[480:1545, 10:750]
#     sector9 = paper[480:1545, 800:1540]
#     sector10 = paper[480:1545, 1590:2505]
#     sector11 = paper[1565: 3350, 5:1150]
#     sector12 = paper[1565: 3350, 1195:2350]
#
#     sector11_part1 = paper[1865: 3350, 5:580]
#     sector11_part2 = paper[1865: 3350, 590:1150]
#
#     sector12_part1 = paper[1865: 3350, 1190:1772]
#     sector12_part2 = paper[1865: 3350, 1772:2350]
#
#     # show_small_image("Sector 8: ", sector8)
#     # show_small_image("Sector 9: ", sector9)
#     # show_small_image("Sector 10: ", sector10)
#     # show_small_image("Sector 11 part 1: ", sector11_part1)
#     # show_small_image("Sector 11 part 2: ", sector11_part2)
#     # show_small_image("Sector 12 part 1: ", sector12_part1)
#     # show_small_image("Sector 12 part 2: ", sector12_part2)
#
#     # sector10_partCircleArea = paper[490:1545, 1590:2500]
#     sector11_bp_selection1 = paper[1650:1975, 300:470]
#     sector11_bp_selection2 = paper[1650:1975, 670:750]
#     sector11_bp_selection3 = paper[1650:1975, 970:1150]
#
#     sector12_bp_selection1 = paper[1650:1975, 1500:1650]
#     sector12_bp_selection2 = paper[1650:1975, 1835:1995]
#     sector12_bp_selection3 = paper[1650:1975, 2195:2350]
#
#     show_small_image("Sector 11 Selection1: ", sector11_bp_selection1)
#     show_small_image("Sector 11 Selection2: ", sector11_bp_selection2)
#     show_small_image("Sector 11 Selection3: ", sector11_bp_selection3)
#     show_small_image("Sector 12 Selection1: ", sector12_bp_selection1)
#     show_small_image("Sector 12 Selection2: ", sector12_bp_selection2)
#     show_small_image("Sector 12 Selection3: ", sector12_bp_selection3)
#
#     circle_area8, circle_area8_warped = finding_the_circle_area(sector8)
#     circle_area9, circle_area9_warped = finding_the_circle_area(sector9)
#     circle_area10, circle_area10_warped = finding_the_circle_area(sector10)
#     circle_area11_part1, circle_area11_part1_warped = finding_the_circle_area(sector11_part1)
#     circle_area11_part2, circle_area11_part2_warped = finding_the_circle_area(sector11_part2)
#     circle_area12_part1, circle_area12_part1_warped = finding_the_circle_area(sector12_part1)
#     circle_area12_part2, circle_area12_part2_warped = finding_the_circle_area(sector12_part2)
#
#     # circle_area11_bp_selection, circle_area11_bp_selection_warped = finding_the_circle_area(sector11_bp_selection)
#     # circle_area12_bp_selection, circle_area12_bp_selection_warped = finding_the_circle_area(sector12_bp_selection)
#     #
#     # show_small_image("Selection part sector 11: ", circle_area11_bp_selection)
#     # show_small_image("Selection part sector 12: ", circle_area12_bp_selection)
#
#     circle_area8 = circle_area8[0: circle_area8.shape[0], 55:circle_area8.shape[1]]
#     circle_area9 = circle_area9[0: circle_area9.shape[0], 55:circle_area9.shape[1]]
#     circle_area10 = circle_area10[0: circle_area10.shape[0], 55:circle_area10.shape[1]]
#     circle_area11_part1 = circle_area11_part1[0: circle_area11_part1.shape[0], 55:circle_area11_part1.shape[1]]
#     circle_area11_part2 = circle_area11_part2[0: circle_area11_part2.shape[0], 55:circle_area11_part2.shape[1]]
#     circle_area12_part1 = circle_area12_part1[0: circle_area12_part1.shape[0], 55:circle_area12_part1.shape[1]]
#     circle_area12_part2 = circle_area12_part2[0: circle_area12_part2.shape[0], 55:circle_area12_part2.shape[1]]
#
#     # show_small_image("CIRCLE AREA 8: ", circle_area8)
#     # circle_area10, circle_area10_warped = finding_the_circle_area(sector10_partCircleArea)
#
#     # show_small_image("Circle area 12 part 1: ", circle_area12_part1)
#     # show_small_image("Circle area 12 part 2: ", circle_area12_part2)
#
#
#     # Append all circle areas to the list
#     list_of_circle_area = []
#     list_of_circle_area.append(circle_area8)
#     list_of_circle_area.append(circle_area9)
#     list_of_circle_area.append(circle_area10)
#     list_of_circle_area.append(circle_area11_part1)
#     list_of_circle_area.append(circle_area11_part2)
#     list_of_circle_area.append(circle_area12_part1)
#     list_of_circle_area.append(circle_area12_part2)
#
#     list_of_circle_area.append(sector11_bp_selection1)
#     list_of_circle_area.append(sector11_bp_selection2)
#     list_of_circle_area.append(sector11_bp_selection3)
#     list_of_circle_area.append(sector12_bp_selection1)
#     list_of_circle_area.append(sector12_bp_selection2)
#     list_of_circle_area.append(sector12_bp_selection3)
#
#     # Convert all .jpg files into .png files and save. Hough Circles function work with .png files
#     c = 1
#     for i in range(len(list_of_circle_area)):
#         cv2.imwrite('images/png_format_circle_area_paper2/circle_area%d.png'%(i), list_of_circle_area[i])
#         c += 1
#         if i == len(list_of_circle_area) - 1:
#             break
#
#     # Reading the .png files
#     circle_area8_png = cv2.imread("images/png_format_circle_area_paper2/circle_area0.png", 0)
#     circle_area9_png = cv2.imread("images/png_format_circle_area_paper2/circle_area1.png", 0)
#     circle_area10_png= cv2.imread("images/png_format_circle_area_paper2/circle_area2.png", 0)
#     circle_area11_part1_png = cv2.imread("images/png_format_circle_area_paper2/circle_area3.png", 0)
#     circle_area11_part2_png = cv2.imread("images/png_format_circle_area_paper2/circle_area4.png", 0)
#     circle_area12_part1_png = cv2.imread("images/png_format_circle_area_paper2/circle_area5.png", 0)
#     circle_area12_part2_png = cv2.imread("images/png_format_circle_area_paper2/circle_area6.png", 0)
#
#     circle_area11_bp_selection1_png = cv2.imread("images/png_format_circle_area_paper2/circle_area7.png", 0)
#     circle_area11_bp_selection2_png = cv2.imread("images/png_format_circle_area_paper2/circle_area8.png", 0)
#     circle_area11_bp_selection3_png = cv2.imread("images/png_format_circle_area_paper2/circle_area9.png", 0)
#     circle_area12_bp_selection1_png = cv2.imread("images/png_format_circle_area_paper2/circle_area10.png", 0)
#     circle_area12_bp_selection2_png = cv2.imread("images/png_format_circle_area_paper2/circle_area11.png", 0)
#     circle_area12_bp_selection3_png = cv2.imread("images/png_format_circle_area_paper2/circle_area12.png", 0)
#
#     # show_small_image("circle area 8 png: ", circle_area8)
#     # Finding the circles
#     circles_s8 = finding_the_circles_by_Hough_Circles(circle_area8_png)
#     circles_s9 = finding_the_circles_by_Hough_Circles(circle_area9_png)
#     circles_s10 = finding_the_circles_by_Hough_Circles(circle_area10_png)
#     circles_s11_p1 = finding_the_circles_by_Hough_Circles(circle_area11_part1_png)
#     circles_s11_p2 = finding_the_circles_by_Hough_Circles(circle_area11_part2_png)
#     circles_s12_p1 = finding_the_circles_by_Hough_Circles(circle_area12_part1_png)
#     circles_s12_p2 = finding_the_circles_by_Hough_Circles(circle_area12_part2_png)
#     circles_s11_bp_selection1 = finding_the_circles_by_Hough_Circles(circle_area11_bp_selection1_png)
#     circles_s11_bp_selection2 = finding_the_circles_by_Hough_Circles(circle_area11_bp_selection2_png)
#     circles_s11_bp_selection3 = finding_the_circles_by_Hough_Circles(circle_area11_bp_selection3_png)
#     circles_s12_bp_selection1 = finding_the_circles_by_Hough_Circles(circle_area12_bp_selection1_png)
#     circles_s12_bp_selection2 = finding_the_circles_by_Hough_Circles(circle_area12_bp_selection2_png)
#     circles_s12_bp_selection3 = finding_the_circles_by_Hough_Circles(circle_area12_bp_selection3_png)
#
#     # show_small_image("Circles: ", circles_s11_bp_selection)
#
#     # Sorting the circles by X or by Y
#     circles_s8 = sort_the_circles_ByY(circles_s8)
#     circles_s9 = sort_the_circles_ByY(circles_s9)
#     circles_s10 = sort_the_circles_ByY(circles_s10)
#     circles_s11_p1 = sort_the_circles_ByY(circles_s11_p1)
#     circles_s11_p2 = sort_the_circles_ByY(circles_s11_p2)
#     circles_s12_p1 = sort_the_circles_ByY(circles_s12_p1)
#     circles_s12_p2 = sort_the_circles_ByY(circles_s12_p2)
#
#     circles_s11_bp_select1 = sort_the_circles_ByX(circles_s11_bp_selection1)
#     circles_s11_bp_select2 = sort_the_circles_ByX(circles_s11_bp_selection2)
#     circles_s11_bp_select3 = sort_the_circles_ByX(circles_s11_bp_selection3)
#     circles_s12_bp_select1 = sort_the_circles_ByX(circles_s12_bp_selection1)
#     circles_s12_bp_select2 = sort_the_circles_ByX(circles_s12_bp_selection2)
#     circles_s12_bp_select3 = sort_the_circles_ByX(circles_s12_bp_selection3)
#
#     # Sorting per row or per column
#     sorted_c_a8 = sort_per_row(circles_s8, numberOfColumns=5)
#     sorted_c_a9 = sort_per_row(circles_s9, numberOfColumns=5)
#     sorted_c_a10 = sort_per_row(circles_s10, numberOfColumns=5)
#     sorted_c_a11_p1 = sort_per_row(circles_s11_p1, numberOfColumns=5)
#     sorted_c_a12_p1 = sort_per_row(circles_s12_p1, numberOfColumns=5)
#     sorted_c_a11_p2 = sort_per_row(circles_s11_p2, numberOfColumns=8)
#     sorted_c_a12_p2 = sort_per_row(circles_s12_p2, numberOfColumns=8)
#
#     sorted_c_s11_bp_selec1 = sort_per_column(circles_s11_bp_select1, numberOfRows=4)
#     sorted_c_s11_bp_selec2 = sort_per_column(circles_s11_bp_select2, numberOfRows=4)
#     sorted_c_s11_bp_selec3 = sort_per_column(circles_s11_bp_select3, numberOfRows=4)
#     sorted_c_s12_bp_selec1 = sort_per_column(circles_s12_bp_select1, numberOfRows=4)
#     sorted_c_s12_bp_selec2 = sort_per_column(circles_s12_bp_select2, numberOfRows=4)
#     sorted_c_s12_bp_selec3 = sort_per_column(circles_s12_bp_select3, numberOfRows=4)
#
#     # Align circles with letters
#     alignedCircWithLettList_s8 = align_circles_with_letters(sorted_c_a8, LETTERS)
#     alignedCircWithLettList_s9 = align_circles_with_letters(sorted_c_a9, LETTERS)
#     alignedCircWithLettList_s10 = align_circles_with_letters(sorted_c_a10, LETTERS)
#     alignedCircWithLettList_s11_p1 = align_circles_with_letters(sorted_c_a11_p1, LETTERS)
#     alignedCircWithLettList_s12_p1 = align_circles_with_letters(sorted_c_a12_p1, LETTERS)
#     alignedCircWithLettList_s11_p2 = align_circles_with_letters(sorted_c_a11_p2, LETTERS2)
#     alignedCircWithLettList_s12_p2 = align_circles_with_letters(sorted_c_a12_p2, LETTERS2)
#
#     alignedCircWithSubjList_s11_bp_s1 = align_circles_with_letters(sorted_c_s11_bp_selec1, SUBJECTS1)
#     alignedCircWithSubjList_s11_bp_s2= align_circles_with_letters(sorted_c_s11_bp_selec2, SUBJECTS2)
#     alignedCircWithSubjList_s11_bp_s3 = align_circles_with_letters(sorted_c_s11_bp_selec3, SUBJECTS3)
#     alignedCircWithSubjList_s12_bp_s1 = align_circles_with_letters(sorted_c_s12_bp_selec1, SUBJECTS1)
#     alignedCircWithSubjList_s12_bp_s2 = align_circles_with_letters(sorted_c_s12_bp_selec2, SUBJECTS2)
#     alignedCircWithSubjList_s12_bp_s3 = align_circles_with_letters(sorted_c_s12_bp_selec3, SUBJECTS3)
#
#     # Finding the bubbled keypoints by Blob Detector function
#     im_with_keypoints8, keypoints8 = finding_the_bubbled_by_BlobDetector(circle_area8)
#     im_with_keypoints9, keypoints9 = finding_the_bubbled_by_BlobDetector(circle_area9)
#     im_with_keypoints10, keypoints10 = finding_the_bubbled_by_BlobDetector(circle_area10)
#     im_with_keypoints11_p1, keypoints11_p1 = finding_the_bubbled_by_BlobDetector(circle_area11_part1)
#     im_with_keypoints11_p2, keypoints11_p2 = finding_the_bubbled_by_BlobDetector(circle_area11_part2)
#     im_with_keypoints12_p1, keypoints12_p1 = finding_the_bubbled_by_BlobDetector(circle_area12_part1)
#     im_with_keypoints12_p2, keypoints12_p2 = finding_the_bubbled_by_BlobDetector(circle_area12_part2)
#
#     im_with_keypoints11_bp_s1, keypoints11_bp_s1 = finding_the_bubbled_by_BlobDetector(sector11_bp_selection1)
#     im_with_keypoints11_bp_s2, keypoints11_bp_s2 = finding_the_bubbled_by_BlobDetector(sector11_bp_selection2)
#     im_with_keypoints11_bp_s3, keypoints11_bp_s3 = finding_the_bubbled_by_BlobDetector(sector11_bp_selection3)
#     im_with_keypoints12_bp_s1, keypoints12_bp_s1 = finding_the_bubbled_by_BlobDetector(sector12_bp_selection1)
#     im_with_keypoints12_bp_s2, keypoints12_bp_s2 = finding_the_bubbled_by_BlobDetector(sector12_bp_selection2)
#     im_with_keypoints12_bp_s3, keypoints12_bp_s3 = finding_the_bubbled_by_BlobDetector(sector12_bp_selection3)
#
#
#     # Sorting keypoints by X
#     sorted_xy_keypoints_s8 = sorting_keypoints_by_Y(keypoints8)
#     sorted_xy_keypoints_s9 = sorting_keypoints_by_Y(keypoints9)
#     sorted_xy_keypoints_s10 = sorting_keypoints_by_Y(keypoints10)
#     sorted_xy_keypoints_s11_p1 = sorting_keypoints_by_Y(keypoints11_p1)
#     sorted_xy_keypoints_s11_p2 = sorting_keypoints_by_Y(keypoints11_p2)
#     sorted_xy_keypoints_s12_p1 = sorting_keypoints_by_Y(keypoints12_p1)
#     sorted_xy_keypoints_s12_p2 = sorting_keypoints_by_Y(keypoints12_p2)
#
#     sorted_xy_keypoints_s11_bp_s1 = sorting_keypoints_by_Y(keypoints11_bp_s1)
#     sorted_xy_keypoints_s11_bp_s2 = sorting_keypoints_by_Y(keypoints11_bp_s2)
#     sorted_xy_keypoints_s11_bp_s3 = sorting_keypoints_by_Y(keypoints11_bp_s3)
#     sorted_xy_keypoints_s12_bp_s1 = sorting_keypoints_by_Y(keypoints12_bp_s1)
#     sorted_xy_keypoints_s12_bp_s2 = sorting_keypoints_by_Y(keypoints12_bp_s2)
#     sorted_xy_keypoints_s12_bp_s3 = sorting_keypoints_by_Y(keypoints12_bp_s3)
#
#     # Print keypoints
#     show_keypoints(sorted_xy_keypoints_s8)
#     show_keypoints(sorted_xy_keypoints_s9)
#     show_keypoints(sorted_xy_keypoints_s10)
#     show_keypoints(sorted_xy_keypoints_s11_p1)
#     show_keypoints(sorted_xy_keypoints_s11_p2)
#     show_keypoints(sorted_xy_keypoints_s12_p1)
#     show_keypoints(sorted_xy_keypoints_s12_p2)
#
#     show_keypoints(sorted_xy_keypoints_s11_bp_s1)
#     show_keypoints(sorted_xy_keypoints_s11_bp_s2)
#     show_keypoints(sorted_xy_keypoints_s11_bp_s3)
#     show_keypoints(sorted_xy_keypoints_s12_bp_s1)
#     show_keypoints(sorted_xy_keypoints_s12_bp_s2)
#     show_keypoints(sorted_xy_keypoints_s12_bp_s3)
#
#
#     characters_s8 = finding_the_bubbled_characters(sorted_xy_keypoints_s8, alignedCircWithLettList_s8)
#     characters_s9 = finding_the_bubbled_characters(sorted_xy_keypoints_s9, alignedCircWithLettList_s9)
#     characters_s10 = finding_the_bubbled_characters(sorted_xy_keypoints_s10, alignedCircWithLettList_s10)
#     characters_s11_p1 = finding_the_bubbled_characters(sorted_xy_keypoints_s11_p1, alignedCircWithLettList_s11_p1)
#     characters_s11_p2 = finding_the_bubbled_characters(sorted_xy_keypoints_s11_p2, alignedCircWithLettList_s11_p2)
#     characters_s12_p1 = finding_the_bubbled_characters(sorted_xy_keypoints_s12_p1, alignedCircWithLettList_s12_p1)
#     characters_s12_p2 = finding_the_bubbled_characters(sorted_xy_keypoints_s12_p2, alignedCircWithLettList_s12_p2)
#
#     characters_s11_bp_s1 = finding_the_bubbled_characters(sorted_xy_keypoints_s11_bp_s1, alignedCircWithSubjList_s11_bp_s1)
#     characters_s11_bp_s2 = finding_the_bubbled_characters(sorted_xy_keypoints_s11_bp_s2, alignedCircWithSubjList_s11_bp_s2)
#     characters_s11_bp_s3 = finding_the_bubbled_characters(sorted_xy_keypoints_s11_bp_s3, alignedCircWithSubjList_s11_bp_s3)
#     characters_s12_bp_s1 = finding_the_bubbled_characters(sorted_xy_keypoints_s12_bp_s1, alignedCircWithSubjList_s12_bp_s1)
#     characters_s12_bp_s2 = finding_the_bubbled_characters(sorted_xy_keypoints_s12_bp_s2, alignedCircWithSubjList_s12_bp_s2)
#     characters_s12_bp_s3 = finding_the_bubbled_characters(sorted_xy_keypoints_s12_bp_s3, alignedCircWithSubjList_s12_bp_s3)
#
#     correct = 0
#     uncorrect = 0
#
#     for i in range(len(characters_s8)-1):
#         if characters_s8[i] == ANSWER_KEY[i]:
#             # print(characters_s8[i])
#             correct += 1
#         else:
#             uncorrect += 1
#
#     print(correct)
#     print(uncorrect)
#
#
#     score = float((correct/20.0) * 100)
#
#     print(characters_s8)
#     print(score)
#     print(characters_s9)
#     print(characters_s10)
#     print(characters_s11_p1)
#     print(characters_s11_p2)
#     print(characters_s12_p1)
#     print(characters_s12_p2)
#
#     print(characters_s11_bp_s1)
#     print(characters_s11_bp_s2)
#     print(characters_s11_bp_s3)
#
#     print(characters_s12_bp_s1)
#     print(characters_s12_bp_s2)
#     print(characters_s12_bp_s3)
#
# # circles_s1 = sort_the_circles_ByX(circles_s1)
# # circles_s2 = sort_the_circles_ByX(circles_s2)
# # circles_s3 = sort_the_circles_ByX(circles_s3)
# # circles_s4 = sort_the_circles_ByX(circles_s4)
#
# if __name__ == "__main__":
#     main()
