import sys, time
from scanning_test_p1_class import EnginePart
from scanning_test_p2_class import EnginePart2
import cv2
import numpy as np
import pandas as pd
import imutils
import math
import xlsxwriter
import xlrd

from imutils import contours
from imutils.perspective import four_point_transform
from openpyxl import load_workbook
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import QIcon, QColor, QImage
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import *
from PyQt5.uic import loadUi


class ScanningTestApp(QMainWindow):
    def __init__(self):
        super(ScanningTestApp, self).__init__()
        self.init_ui()
        loadUi("scanning_test.ui", self)

        self.original_image = None
        self.original2_image = None

        self.engP = EnginePart()
        self.engP2 = EnginePart2()

        # Data frames
        self.df = None
        self.df2 = None
        self.df3 = None

        # Applicants marked data
        self.characters_s1 = None
        self.characters_s2 = None
        self.characters_s3 = None
        self.characters_s4 = None
        self.characters_s5 = None
        self.characters_s6 = None

        self.characters_s8 = None
        self.characters_s9 = None
        self.characters_s10 = None

        self.characters_s11_p1 = None
        self.characters_s11_p2 = None
        self.characters_s12_p1 = None
        self.characters_s12_p2 = None

        self.characters_s11_bp_c1 = None
        self.characters_s11_bp_c2 = None
        self.characters_s11_bp_c3 = None
        self.characters_s12_bp_c1 = None
        self.characters_s12_bp_c2 = None
        self.characters_s12_bp_c3 = None

        self.characters_11_bp_selection = None
        self.characters_12_bp_selection = None

        self.kz_alphabet = np.array(['А', 'Ә', 'Б', 'В',
                                     'Г', 'Ғ', 'Д', 'Е',
                                     'Ж', 'З', 'И', 'Й',
                                     'К', 'Қ', 'Л', 'М',
                                     'Н', 'Ң', 'О', 'Ө',
                                     'П', 'Р', 'С', 'Т',
                                     'У', 'Ұ', 'Ү', 'Ф',
                                     'Х', 'Һ', 'Ц', 'Ч',
                                     'Ш', 'Щ', 'Ь', 'Ы',
                                     'І', 'Э', 'Ю', 'Я',
                                     '-'])

        self.numbers_for_s3_s4 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])

        self.kz_alphabet_for_s5 = np.array(['А', 'Ә', 'Б', 'В',
                                            'Г', 'Ғ', 'Д', 'Е',
                                            'Ж', 'З', 'И', 'К',
                                            'Л', 'М', 'Н', 'О',
                                            'П', 'Р', 'С', 'У',
                                            'Ф', 'Х', 'Э', 'Ю'])

        self.numbers_for_s6 = np.array([1, 2, 3, 4, 5,
                                        6, 7, 8, 9, 10,
                                        11, 12, 13, 14,
                                        15, 16, 17, 18,
                                        19, 20, 21, 22,
                                        23, 24])

        self.LETTERS = np.array(['A', 'B', 'C', 'D', 'E'])

        self.LETTERS2 = np.array(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'])

        self.SUBJECTS1 = np.array(['Қазақ тілі және әдебиет',
                                   'Орыс тілі және әдебиет',
                                   'Адам. Қоғам. Құқық.',
                                   'Математика'])

        self.SUBJECTS2 = np.array(['Химия',
                                   'Физика',
                                   'Биология',
                                   'Тарих'])

        self.SUBJECTS3 = np.array(['География',
                                   'Ағылшын тілі',
                                   'Неміс тілі',
                                   'Француз тілі'])

        self.timer = QBasicTimer()
        self.step = 0

        self.loadImageBtn.clicked.connect(self.loadBtnClicked)
        self.scanBtn.clicked.connect(self.scanTest)
        self.loadFileWithAnswersBtn.clicked.connect(self.loadFileWithAnswers)


    def init_ui(self):

        window_width = 1500
        window_height = 900

        # Create widget
        background_image = QLabel(self)
        pixmap = QPixmap('images/backgrounds/fedora2.jpg')
        pixmap = pixmap.scaled(window_width, window_height)
        background_image.setPixmap(pixmap)
        background_image.resize(window_width, window_height)

        self.pbar = QProgressBar()

        self.setMaximumSize(window_width, window_height)
        self.setWindowTitle('Scanning test app')
        self.setWindowIcon(QIcon('212567.png'))


    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())


    @pyqtSlot()
    def loadBtnClicked(self):
        fileNames = QFileDialog().getOpenFileNames()
        # print(fileNames)

        if len(fileNames) == 2:
            filePathes = fileNames[0]
            if len(filePathes) == 2:
                self.loadImage(filePathes)
            else:
                print("Invalid Image!")


    def loadImage(self, filePathes):
        self.original_image = cv2.imread(filePathes[0])
        self.original2_image = cv2.imread(filePathes[1])
        self.engP.displayImage(self.original_image, self.imgLabel)
        self.engP.displayImage(self.original2_image, self.img2Label2)
        # self.imgLabel.setGeometry(0,0,800,800)


    def timerEvent(self, event):
        if self.step >= 100:
            self.timer.stop()
            return
        self.step = self.step + 1
        self.pbar.setValue(self.step)


    @pyqtSlot()
    def scanTest(self):
        self.timer.start(100, self)
        self.df = pd.read_excel('grades.xlsx')
        self.df3 = pd.read_excel('applicants_marked_data.xlsx')

        if self.original_image is not None:

            black_squares = self.engP.finding_the_black_squares(self.original_image)

            print(len(black_squares))
            # if len(black_squares) == 0:

            four_points = self.engP.check_for_four_points(black_squares, self.original_image)

            for i in range(len(black_squares)):
                cv2.drawContours(self.original_image, black_squares[i], -1, (240, 200, 100), 15)
                self.engP.displayImage(self.original_image, self.imgLabel)
                # show_small_image("Black squares: ", original_image)
            paper = self.engP.align_by_four_points(four_points, self.original_image)

            # self.engP.displayImage(paper, self.imgLabel)

            sectors = self.engP.finding_sectors(paper)
            sectors = sorted(sectors, key=cv2.contourArea, reverse=True)
            # cv2.drawContours(self.original_image, sectors, -1, (240, 200, 100), 15)
            # self.engP.displayImage(self.original_image, self.imgLabel)

            # Finding sectors
            sector1 = paper[2345:3360, 0:2185]
            sector2 = paper[1305:2355, 0:2185]
            sector3 = paper[565:1295, 600:1140]
            sector4 = paper[175:485, 600:1140]
            sector5 = paper[925:1315, 1300:1520]
            sector6 = paper[145:515, 1300:1520]
            # sector7 = four_point_transform(paper, sectors[4].reshape(4, 2))

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
            # rotated_sector7 = imutils.rotate_bound(sector7, 90)

            # Finding circle area
            circle_area1, circle_area1_warped = self.engP.finding_the_circle_area(rotated_sector1)
            circle_area2, circle_area2_warped = self.engP.finding_the_circle_area(rotated_sector2)
            circle_area3, circle_area3_warped = self.engP.finding_the_circle_area(rotated_sector3)
            circle_area4, circle_area4_warped = self.engP.finding_the_circle_area(rotated_sector4)
            circle_area5, circle_area5_warped = self.engP.finding_the_circle_area(rotated_sector5)
            circle_area6, circle_area6_warped = self.engP.finding_the_circle_area(rotated_sector6)

            # Append all circle areas to the list
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
                cv2.imwrite('images/png_format_circle_area/circle_area%d.png' % (c), list_of_circle_area[i])
                c += 1
                if i == len(list_of_circle_area) - 1:
                    break

            # Reading the .png files
            circle_area1_png = cv2.imread("images/png_format_circle_area/circle_area1.png", 0)
            circle_area2_png = cv2.imread("images/png_format_circle_area/circle_area2.png", 0)
            circle_area3_png = cv2.imread("images/png_format_circle_area/circle_area3.png", 0)
            circle_area4_png = cv2.imread("images/png_format_circle_area/circle_area4.png", 0)
            circle_area5_png = cv2.imread("images/png_format_circle_area/circle_area5.png", 0)
            circle_area6_png = cv2.imread("images/png_format_circle_area/circle_area6.png", 0)

            # Finding the circles
            circles_s1 = self.engP.finding_the_circles_by_Hough_Circles(circle_area1_png)
            circles_s2 = self.engP.finding_the_circles_by_Hough_Circles(circle_area2_png)
            circles_s3 = self.engP.finding_the_circles_by_Hough_Circles(circle_area3_png)
            circles_s4 = self.engP.finding_the_circles_by_Hough_Circles(circle_area4_png)
            circles_s5 = self.engP.finding_the_circles_by_Hough_Circles(circle_area5_png)
            circles_s6 = self.engP.finding_the_circles_by_Hough_Circles(circle_area6_png)

            # for i in circles_s1[0, :]:
            #     cv2.circle(circle_area1, (i[0], i[1]), i[2], (244, 128, 66), 10)
            #     cv2.circle(circle_area1, (i[0], i[1]), 2, (200, 0, 0), 5)

            # cv2.drawContours(self.original_image, circles_s1[0], -1, (255, 0, 0), 10)
            # cv2.drawContours(self.original_image, circles_s2, -1, (255, 0, 0), 10)
            # self.engP.displayImage(circle_area1, self.imgLabel)

            # Sorting the circles by X or by Y
            circles_s1 = self.engP.sort_the_circles_ByX(circles_s1)
            circles_s2 = self.engP.sort_the_circles_ByX(circles_s2)
            circles_s3 = self.engP.sort_the_circles_ByX(circles_s3)
            circles_s4 = self.engP.sort_the_circles_ByX(circles_s4)
            circles_s5 = self.engP.sort_the_circles_ByY(circles_s5)
            circles_s6 = self.engP.sort_the_circles_ByY(circles_s6)

            # Sorting per row or per column
            sorted_c_a1 = self.engP.sort_per_column(circles_s1, numberOfRows=41)
            sorted_c_a2 = self.engP.sort_per_column(circles_s2, numberOfRows=41)
            sorted_c_a3 = self.engP.sort_per_column(circles_s3, numberOfRows=10)
            sorted_c_a4 = self.engP.sort_per_column(circles_s4, numberOfRows=10)
            sorted_c_a5 = self.engP.sort_per_row(circles_s5, numberOfColumns=6)
            sorted_c_a6 = self.engP.sort_per_row(circles_s6, numberOfColumns=6)


            # Align circles with letters
            alignedCircWithLettList_s1 = self.engP.align_circles_with_letters(sorted_c_a1, self.kz_alphabet)
            alignedCircWithLettList_s2 = self.engP.align_circles_with_letters(sorted_c_a2, self.kz_alphabet)
            alignedCircWithNumbList_s3 = self.engP.align_circles_with_letters(sorted_c_a3, self.numbers_for_s3_s4)
            alignedCircWithNumbList_s4 = self.engP.align_circles_with_letters(sorted_c_a4, self.numbers_for_s3_s4)
            alignedCircWithLettList_s5 = self.engP.align_circles_with_letters(sorted_c_a5, self.kz_alphabet_for_s5)
            alignedCircWithLettList_s6 = self.engP.align_circles_with_letters(sorted_c_a6, self.numbers_for_s6)

            # Finding the bubbled keypoints by Blob Detector function
            im_with_keypoints1, keypoints1 = self.engP.finding_the_bubbled_by_BlobDetector(circle_area1)
            im_with_keypoints2, keypoints2 = self.engP.finding_the_bubbled_by_BlobDetector(circle_area2)
            im_with_keypoints3, keypoints3 = self.engP.finding_the_bubbled_by_BlobDetector(circle_area3)
            im_with_keypoints4, keypoints4 = self.engP.finding_the_bubbled_by_BlobDetector(circle_area4)
            im_with_keypoints5, keypoints5 = self.engP.finding_the_bubbled_by_BlobDetector(circle_area5)
            im_with_keypoints6, keypoints6 = self.engP.finding_the_bubbled_by_BlobDetector(circle_area6)
            # im_with_keypoints7, keypoints7 = self.engP.finding_the_bubbled_by_BlobDetector(circle_area_s7)

            # Sorting keypoints by X
            sorted_xy_keypoints_s1 = self.engP.sorting_keypoints_by_X(keypoints1)
            sorted_xy_keypoints_s2 = self.engP.sorting_keypoints_by_X(keypoints2)
            sorted_xy_keypoints_s3 = self.engP.sorting_keypoints_by_X(keypoints3)
            sorted_xy_keypoints_s4 = self.engP.sorting_keypoints_by_X(keypoints4)
            sorted_xy_keypoints_s5 = self.engP.sorting_keypoints_by_X(keypoints5)
            sorted_xy_keypoints_s6 = self.engP.sorting_keypoints_by_X(keypoints6)

            # Print keypoints
            self.engP.show_keypoints(sorted_xy_keypoints_s1)
            self.engP.show_keypoints(sorted_xy_keypoints_s2)
            self.engP.show_keypoints(sorted_xy_keypoints_s3)
            self.engP.show_keypoints(sorted_xy_keypoints_s4)
            self.engP.show_keypoints(sorted_xy_keypoints_s5)
            self.engP.show_keypoints(sorted_xy_keypoints_s6)

            self.characters_s1 = self.engP.finding_the_bubbled_characters(sorted_xy_keypoints_s1, alignedCircWithLettList_s1)
            self.characters_s2 = self.engP.finding_the_bubbled_characters(sorted_xy_keypoints_s2, alignedCircWithLettList_s2)
            self.characters_s3 = self.engP.finding_the_bubbled_characters(sorted_xy_keypoints_s3, alignedCircWithNumbList_s3)
            self.characters_s4 = self.engP.finding_the_bubbled_characters(sorted_xy_keypoints_s4, alignedCircWithNumbList_s4)
            self.characters_s5 = self.engP.finding_the_bubbled_characters(sorted_xy_keypoints_s5, alignedCircWithLettList_s5)
            self.characters_s6 = self.engP.finding_the_bubbled_characters(sorted_xy_keypoints_s6, alignedCircWithLettList_s6)

            print(self.characters_s1)
            print(self.characters_s2)
            print(self.characters_s3)
            print(self.characters_s4)
            print(self.characters_s5)
            print(self.characters_s6)

            self.df.loc[len(self.df) + 1, ['Тегі-Фамилия', 'Аты-Имя', 'ЖСН-ИИН', 'НҰСҚА-ВАРИАНТ', 'Сынып литерасы-Литера класса', 'Қосымша сектор-Резервный сектор']]\
                = [self.characters_s1[-1], self.characters_s2[-1], self.characters_s3[-1], self.characters_s4[-1], self.characters_s5[-1], self.characters_s6[-1]]
            print("Length of Data Frame: ", len(self.df))

            # characters_s3, characters_s4, characters_s5, characters_s6
            # 'Тегі-Фамилия Аты-Имя ЖСН-ИИН НҰСҚА-ВАРИАНТ Сынып литерасы-Литера класса Қосымша сектор-Резервный сектор Алған балл-ы(Пайызбен %):'

            # workbook = xlsxwriter.Workbook('applicants.xlsx')
            # worksheet = workbook.add_worksheet()
            #
            # # Add a bold format to use to highlight cells.
            # bold = workbook.add_format({'bold': True})
            #
            # # Add a number format for cells with money.
            # money = workbook.add_format({'num_format': '$#,##0'})
            #
            # # Write some data headers.
            # worksheet.write('A1', 'Тегі-Фамилия', bold)
            # worksheet.write('B1', 'Аты-Имя', bold)
            # worksheet.write('C1', 'ЖСН-ИИН', bold)
            # worksheet.write('D1', 'Нұсқа-Вариант', bold)
            # worksheet.write('E1', 'Сынып литерасы-Литера класса', bold)
            # worksheet.write('F1', 'Қосымша сектор-Резервный сектор', bold)
            # worksheet.write('H1', 'Математикалық сауаттылық:', bold)
            #
            # row = 1
            # col = 0
            #
            # worksheet.write(row, col, characters_s1[-1])
            # worksheet.write(row, col + 1, characters_s2[-1])
            # worksheet.write(row, col + 2, characters_s3[-1])
            # worksheet.write(row, col + 3, characters_s4[-1])
            # worksheet.write(row, col + 4, characters_s5[-1])
            # worksheet.write(row, col + 5, characters_s6[-1])
            #
            # workbook.close()

        if self.original2_image is not None:
            print("Second paper is started!")

            ANSWER_KEY = {0: 'B', 1: 'C', 2: 'D', 3: 'E', 4: 'E', 5: 'C', 6: 'D', 7: 'C', 8: 'A', 9: 'A', 10: 'D',
                          11: 'B', 12: 'B', 13: 'D', 14: 'A', 15: 'C', 16: 'D', 17: 'E', 18: 'A', 19: 'D'}

            ANSWER_KEY2 = {0: 'B', 1: 'C', 2: 'D', 3: 'E', 4: 'E', 5: 'C', 6: 'D', 7: 'C', 8: 'A', 9: 'A', 10: 'D',
                          11: 'B', 12: 'B', 13: 'D', 14: 'A', 15: 'C', 16: 'D', 17: 'E', 18: 'A', 19: 'D'}

            black_squares = self.engP2.finding_the_black_squares(self.original2_image)

            print(len(black_squares))
            #
            # for i in range(len(black_squares)):
            #     cv2.drawContours(original2_image, black_squares[i], -1, (200, 240, 100), 15)

            four_points = self.engP2.check_for_four_points(black_squares, self.original2_image)

            for i in range(len(black_squares)):
                cv2.drawContours(self.original_image, black_squares[i], -1, (200, 240, 100), 15)

            paper = self.engP2.align_by_four_points(four_points, self.original2_image)

            sectors = self.engP2.finding_sectors(paper)
            sectors = sorted(sectors, key=cv2.contourArea, reverse=True)

            # Finding sectors
            sector8 = paper[480:1545, 10:750]
            sector9 = paper[480:1545, 800:1540]
            sector10 = paper[480:1545, 1590:2505]
            sector11 = paper[1565: 3350, 5:1150]
            sector12 = paper[1565: 3350, 1195:2350]

            sector11_part1 = paper[1865: 3350, 5:580]
            sector11_part2 = paper[1865: 3350, 590:1150]

            sector12_part1 = paper[1865: 3350, 1190:1772]
            sector12_part2 = paper[1865: 3350, 1772:2350]

            # show_small_image("Sector 8: ", sector8)
            # show_small_image("Sector 9: ", sector9)
            # show_small_image("Sector 10: ", sector10)
            # show_small_image("Sector 11 part 1: ", sector11_part1)
            # show_small_image("Sector 11 part 2: ", sector11_part2)
            # show_small_image("Sector 12 part 1: ", sector12_part1)
            # show_small_image("Sector 12 part 2: ", sector12_part2)

            # sector10_partCircleArea = paper[490:1545, 1590:2500]
            sector11_bp_selection1 = paper[1650:1975, 300:470]
            sector11_bp_selection2 = paper[1650:1975, 670:750]
            sector11_bp_selection3 = paper[1650:1975, 970:1150]

            sector12_bp_selection1 = paper[1650:1975, 1500:1650]
            sector12_bp_selection2 = paper[1650:1975, 1835:1995]
            sector12_bp_selection3 = paper[1650:1975, 2195:2350]

            # self.engP2.show_small_image("Sector 11 Selection1: ", sector11_bp_selection1)
            # self.engP2.show_small_image("Sector 11 Selection2: ", sector11_bp_selection2)
            # self.engP2.show_small_image("Sector 11 Selection3: ", sector11_bp_selection3)
            # self.engP2.show_small_image("Sector 12 Selection1: ", sector12_bp_selection1)
            # self.engP2.show_small_image("Sector 12 Selection2: ", sector12_bp_selection2)
            # self.engP2.show_small_image("Sector 12 Selection3: ", sector12_bp_selection3)

            circle_area8, circle_area8_warped = self.engP2.finding_the_circle_area(sector8)
            circle_area9, circle_area9_warped = self.engP2.finding_the_circle_area(sector9)
            circle_area10, circle_area10_warped = self.engP2.finding_the_circle_area(sector10)
            circle_area11_part1, circle_area11_part1_warped = self.engP2.finding_the_circle_area(sector11_part1)
            circle_area11_part2, circle_area11_part2_warped = self.engP2.finding_the_circle_area(sector11_part2)
            circle_area12_part1, circle_area12_part1_warped = self.engP2.finding_the_circle_area(sector12_part1)
            circle_area12_part2, circle_area12_part2_warped = self.engP2.finding_the_circle_area(sector12_part2)

            # circle_area11_bp_selection, circle_area11_bp_selection_warped = finding_the_circle_area(sector11_bp_selection)
            # circle_area12_bp_selection, circle_area12_bp_selection_warped = finding_the_circle_area(sector12_bp_selection)
            #
            # show_small_image("Selection part sector 11: ", circle_area11_bp_selection)
            # show_small_image("Selection part sector 12: ", circle_area12_bp_selection)

            circle_area8 = circle_area8[0: circle_area8.shape[0], 55:circle_area8.shape[1]]
            circle_area9 = circle_area9[0: circle_area9.shape[0], 55:circle_area9.shape[1]]
            circle_area10 = circle_area10[0: circle_area10.shape[0], 55:circle_area10.shape[1]]
            circle_area11_part1 = circle_area11_part1[0: circle_area11_part1.shape[0], 55:circle_area11_part1.shape[1]]
            circle_area11_part2 = circle_area11_part2[0: circle_area11_part2.shape[0], 55:circle_area11_part2.shape[1]]
            circle_area12_part1 = circle_area12_part1[0: circle_area12_part1.shape[0], 55:circle_area12_part1.shape[1]]
            circle_area12_part2 = circle_area12_part2[0: circle_area12_part2.shape[0], 55:circle_area12_part2.shape[1]]

            # show_small_image("CIRCLE AREA 8: ", circle_area8)
            # circle_area10, circle_area10_warped = finding_the_circle_area(sector10_partCircleArea)

            # show_small_image("Circle area 12 part 1: ", circle_area12_part1)
            # show_small_image("Circle area 12 part 2l_: ", circle_area12_part2)

            # Append all circle areas to the list
            list_of_circle_area = []
            list_of_circle_area.append(circle_area8)
            list_of_circle_area.append(circle_area9)
            list_of_circle_area.append(circle_area10)
            list_of_circle_area.append(circle_area11_part1)
            list_of_circle_area.append(circle_area11_part2)
            list_of_circle_area.append(circle_area12_part1)
            list_of_circle_area.append(circle_area12_part2)

            list_of_circle_area.append(sector11_bp_selection1)
            list_of_circle_area.append(sector11_bp_selection2)
            list_of_circle_area.append(sector11_bp_selection3)
            list_of_circle_area.append(sector12_bp_selection1)
            list_of_circle_area.append(sector12_bp_selection2)
            list_of_circle_area.append(sector12_bp_selection3)

            # Convert all .jpg files into .png files and save. Hough Circles function work with .png files
            c = 1
            for i in range(len(list_of_circle_area)):
                cv2.imwrite('images/png_format_circle_area_paper2/circle_area%d.png' % (i), list_of_circle_area[i])
                c += 1
                if i == len(list_of_circle_area) - 1:
                    break

            # Reading the .png files
            circle_area8_png = cv2.imread("images/png_format_circle_area_paper2/circle_area0.png", 0)
            circle_area9_png = cv2.imread("images/png_format_circle_area_paper2/circle_area1.png", 0)
            circle_area10_png = cv2.imread("images/png_format_circle_area_paper2/circle_area2.png", 0)
            circle_area11_part1_png = cv2.imread("images/png_format_circle_area_paper2/circle_area3.png", 0)
            circle_area11_part2_png = cv2.imread("images/png_format_circle_area_paper2/circle_area4.png", 0)
            circle_area12_part1_png = cv2.imread("images/png_format_circle_area_paper2/circle_area5.png", 0)
            circle_area12_part2_png = cv2.imread("images/png_format_circle_area_paper2/circle_area6.png", 0)

            circle_area11_bp_selection1_png = cv2.imread("images/png_format_circle_area_paper2/circle_area7.png", 0)
            circle_area11_bp_selection2_png = cv2.imread("images/png_format_circle_area_paper2/circle_area8.png", 0)
            circle_area11_bp_selection3_png = cv2.imread("images/png_format_circle_area_paper2/circle_area9.png", 0)
            circle_area12_bp_selection1_png = cv2.imread("images/png_format_circle_area_paper2/circle_area10.png", 0)
            circle_area12_bp_selection2_png = cv2.imread("images/png_format_circle_area_paper2/circle_area11.png", 0)
            circle_area12_bp_selection3_png = cv2.imread("images/png_format_circle_area_paper2/circle_area12.png", 0)

            # Finding the circles
            circles_s8 = self.engP2.finding_the_circles_by_Hough_Circles(circle_area8_png)
            circles_s9 = self.engP2.finding_the_circles_by_Hough_Circles(circle_area9_png)
            circles_s10 = self.engP2.finding_the_circles_by_Hough_Circles(circle_area10_png)

            circles_s11_p1 = self.engP2.finding_the_circles_by_Hough_Circles(circle_area11_part1_png)
            circles_s11_p2 = self.engP2.finding_the_circles_by_Hough_Circles(circle_area11_part2_png)
            circles_s12_p1 = self.engP2.finding_the_circles_by_Hough_Circles(circle_area12_part1_png)
            circles_s12_p2 = self.engP2.finding_the_circles_by_Hough_Circles(circle_area12_part2_png)

            circles_s11_bp_selection1 = self.engP2.finding_the_circles_by_Hough_Circles(circle_area11_bp_selection1_png)
            circles_s11_bp_selection2 = self.engP2.finding_the_circles_by_Hough_Circles(circle_area11_bp_selection2_png)
            circles_s11_bp_selection3 = self.engP2.finding_the_circles_by_Hough_Circles(circle_area11_bp_selection3_png)
            circles_s12_bp_selection1 = self.engP2.finding_the_circles_by_Hough_Circles(circle_area12_bp_selection1_png)
            circles_s12_bp_selection2 = self.engP2.finding_the_circles_by_Hough_Circles(circle_area12_bp_selection2_png)
            circles_s12_bp_selection3 = self.engP2.finding_the_circles_by_Hough_Circles(circle_area12_bp_selection3_png)

            # Sorting the circles by X or by Y
            circles_s8 = self.engP2.sort_the_circles_ByY(circles_s8)
            circles_s9 = self.engP2.sort_the_circles_ByY(circles_s9)
            circles_s10 = self.engP2.sort_the_circles_ByY(circles_s10)

            circles_s11_p1 = self.engP2.sort_the_circles_ByY(circles_s11_p1)
            circles_s11_p2 = self.engP2.sort_the_circles_ByY(circles_s11_p2)
            circles_s12_p1 = self.engP2.sort_the_circles_ByY(circles_s12_p1)
            circles_s12_p2 = self.engP2.sort_the_circles_ByY(circles_s12_p2)

            circles_s11_bp_select1 = self.engP2.sort_the_circles_ByX(circles_s11_bp_selection1)
            circles_s11_bp_select2 = self.engP2.sort_the_circles_ByX(circles_s11_bp_selection2)
            circles_s11_bp_select3 = self.engP2.sort_the_circles_ByX(circles_s11_bp_selection3)
            circles_s12_bp_select1 = self.engP2.sort_the_circles_ByX(circles_s12_bp_selection1)
            circles_s12_bp_select2 = self.engP2.sort_the_circles_ByX(circles_s12_bp_selection2)
            circles_s12_bp_select3 = self.engP2.sort_the_circles_ByX(circles_s12_bp_selection3)

            # Sorting per row or per column
            sorted_c_a8 = self.engP2.sort_per_row(circles_s8, numberOfColumns=5)
            sorted_c_a9 = self.engP2.sort_per_row(circles_s9, numberOfColumns=5)
            sorted_c_a10 = self.engP2.sort_per_row(circles_s10, numberOfColumns=5)
            sorted_c_a11_p1 = self.engP2.sort_per_row(circles_s11_p1, numberOfColumns=5)
            sorted_c_a12_p1 = self.engP2.sort_per_row(circles_s12_p1, numberOfColumns=5)
            sorted_c_a11_p2 = self.engP2.sort_per_row(circles_s11_p2, numberOfColumns=8)
            sorted_c_a12_p2 = self.engP2.sort_per_row(circles_s12_p2, numberOfColumns=8)

            sorted_c_s11_bp_selec1 = self.engP2.sort_per_column(circles_s11_bp_select1, numberOfRows=4)
            sorted_c_s11_bp_selec2 = self.engP2.sort_per_column(circles_s11_bp_select2, numberOfRows=4)
            sorted_c_s11_bp_selec3 = self.engP2.sort_per_column(circles_s11_bp_select3, numberOfRows=4)
            sorted_c_s12_bp_selec1 = self.engP2.sort_per_column(circles_s12_bp_select1, numberOfRows=4)
            sorted_c_s12_bp_selec2 = self.engP2.sort_per_column(circles_s12_bp_select2, numberOfRows=4)
            sorted_c_s12_bp_selec3 = self.engP2.sort_per_column(circles_s12_bp_select3, numberOfRows=4)

            # Align circles with letters
            alignedCircWithLettList_s8 = self.engP2.align_circles_with_letters(sorted_c_a8, self.LETTERS)
            alignedCircWithLettList_s9 = self.engP2.align_circles_with_letters(sorted_c_a9, self.LETTERS)
            alignedCircWithLettList_s10 = self.engP2.align_circles_with_letters(sorted_c_a10, self.LETTERS)
            alignedCircWithLettList_s11_p1 = self.engP2.align_circles_with_letters(sorted_c_a11_p1, self.LETTERS)
            alignedCircWithLettList_s12_p1 = self.engP2.align_circles_with_letters(sorted_c_a12_p1, self.LETTERS)
            alignedCircWithLettList_s11_p2 = self.engP2.align_circles_with_letters(sorted_c_a11_p2, self.LETTERS2)
            alignedCircWithLettList_s12_p2 = self.engP2.align_circles_with_letters(sorted_c_a12_p2, self.LETTERS2)

            alignedCircWithSubjList_s11_bp_s1 = self.engP2.align_circles_with_letters(sorted_c_s11_bp_selec1, self.SUBJECTS1)
            alignedCircWithSubjList_s11_bp_s2 = self.engP2.align_circles_with_letters(sorted_c_s11_bp_selec2, self.SUBJECTS2)
            alignedCircWithSubjList_s11_bp_s3 = self.engP2.align_circles_with_letters(sorted_c_s11_bp_selec3, self.SUBJECTS3)
            alignedCircWithSubjList_s12_bp_s1 = self.engP2.align_circles_with_letters(sorted_c_s12_bp_selec1, self.SUBJECTS1)
            alignedCircWithSubjList_s12_bp_s2 = self.engP2.align_circles_with_letters(sorted_c_s12_bp_selec2, self.SUBJECTS2)
            alignedCircWithSubjList_s12_bp_s3 = self.engP2.align_circles_with_letters(sorted_c_s12_bp_selec3, self.SUBJECTS3)

            # Finding the bubbled keypoints by Blob Detector function
            im_with_keypoints8, keypoints8 = self.engP2.finding_the_bubbled_by_BlobDetector(circle_area8)
            im_with_keypoints9, keypoints9 = self.engP2.finding_the_bubbled_by_BlobDetector(circle_area9)
            im_with_keypoints10, keypoints10 = self.engP2.finding_the_bubbled_by_BlobDetector(circle_area10)
            im_with_keypoints11_p1, keypoints11_p1 = self.engP2.finding_the_bubbled_by_BlobDetector(circle_area11_part1)
            im_with_keypoints11_p2, keypoints11_p2 = self.engP2.finding_the_bubbled_by_BlobDetector(circle_area11_part2)
            im_with_keypoints12_p1, keypoints12_p1 = self.engP2.finding_the_bubbled_by_BlobDetector(circle_area12_part1)
            im_with_keypoints12_p2, keypoints12_p2 = self.engP2.finding_the_bubbled_by_BlobDetector(circle_area12_part2)


            im_with_keypoints11_bp_s1, keypoints11_bp_s1 = self.engP2.finding_the_bubbled_by_BlobDetector(sector11_bp_selection1)
            im_with_keypoints11_bp_s2, keypoints11_bp_s2 = self.engP2.finding_the_bubbled_by_BlobDetector(sector11_bp_selection2)
            im_with_keypoints11_bp_s3, keypoints11_bp_s3 = self.engP2.finding_the_bubbled_by_BlobDetector(sector11_bp_selection3)
            im_with_keypoints12_bp_s1, keypoints12_bp_s1 = self.engP2.finding_the_bubbled_by_BlobDetector(sector12_bp_selection1)
            im_with_keypoints12_bp_s2, keypoints12_bp_s2 = self.engP2.finding_the_bubbled_by_BlobDetector(sector12_bp_selection2)
            im_with_keypoints12_bp_s3, keypoints12_bp_s3 = self.engP2.finding_the_bubbled_by_BlobDetector(sector12_bp_selection3)

            # Sorting keypoints by Y
            sorted_xy_keypoints_s8 = self.engP2.sorting_keypoints_by_Y(keypoints8)
            sorted_xy_keypoints_s9 = self.engP2.sorting_keypoints_by_Y(keypoints9)
            sorted_xy_keypoints_s10 = self.engP2.sorting_keypoints_by_Y(keypoints10)

            sorted_xy_keypoints_s11_p1 = self.engP2.sorting_keypoints_by_Y(keypoints11_p1)
            # print(sorted_xy_keypoints_s11_p1)
            # sorted_xy_keypoints_s11_p1 = self.engP2.sorting_keypoints_by_X(sorted_xy_keypoints_s11_p1)

            sorted_xy_keypoints_s11_p2 = self.engP2.sorting_keypoints_by_Y(keypoints11_p2)
            sorted_xy_keypoints_s12_p1 = self.engP2.sorting_keypoints_by_Y(keypoints12_p1)
            sorted_xy_keypoints_s12_p2 = self.engP2.sorting_keypoints_by_Y(keypoints12_p2)

            sorted_xy_keypoints_s11_bp_s1 = self.engP2.sorting_keypoints_by_Y(keypoints11_bp_s1)
            sorted_xy_keypoints_s11_bp_s2 = self.engP2.sorting_keypoints_by_Y(keypoints11_bp_s2)
            sorted_xy_keypoints_s11_bp_s3 = self.engP2.sorting_keypoints_by_Y(keypoints11_bp_s3)
            sorted_xy_keypoints_s12_bp_s1 = self.engP2.sorting_keypoints_by_Y(keypoints12_bp_s1)
            sorted_xy_keypoints_s12_bp_s2 = self.engP2.sorting_keypoints_by_Y(keypoints12_bp_s2)
            sorted_xy_keypoints_s12_bp_s3 = self.engP2.sorting_keypoints_by_Y(keypoints12_bp_s3)

            # sorted_xy_keypoints_s8 = self.engP2.sort_per_row(keypoints8, numberOfColumns=5)

            # Print keypoints
            self.engP2.show_keypoints(sorted_xy_keypoints_s8)
            self.engP2.show_keypoints(sorted_xy_keypoints_s9)
            self.engP2.show_keypoints(sorted_xy_keypoints_s10)
            self.engP2.show_keypoints(sorted_xy_keypoints_s11_p1)
            self.engP2.show_keypoints(sorted_xy_keypoints_s11_p2)
            self.engP2.show_keypoints(sorted_xy_keypoints_s12_p1)
            self.engP2.show_keypoints(sorted_xy_keypoints_s12_p2)

            self.engP2.show_keypoints(sorted_xy_keypoints_s11_bp_s1)
            self.engP2.show_keypoints(sorted_xy_keypoints_s11_bp_s2)
            self.engP2.show_keypoints(sorted_xy_keypoints_s11_bp_s3)
            self.engP2.show_keypoints(sorted_xy_keypoints_s12_bp_s1)
            self.engP2.show_keypoints(sorted_xy_keypoints_s12_bp_s2)
            self.engP2.show_keypoints(sorted_xy_keypoints_s12_bp_s3)

            self.characters_s8 = self.engP2.finding_the_bubbled_characters(sorted_xy_keypoints_s8, alignedCircWithLettList_s8)
            self.characters_s9 = self.engP2.finding_the_bubbled_characters(sorted_xy_keypoints_s9, alignedCircWithLettList_s9)
            self.characters_s10 = self.engP2.finding_the_bubbled_characters(sorted_xy_keypoints_s10, alignedCircWithLettList_s10)
            self.characters_s11_p1 = self.engP2.finding_the_bubbled_characters(sorted_xy_keypoints_s11_p1,
                                                               alignedCircWithLettList_s11_p1)
            self.characters_s11_p2 = self.engP2.finding_the_bubbled_characters(sorted_xy_keypoints_s11_p2,
                                                               alignedCircWithLettList_s11_p2)
            self.characters_s12_p1 = self.engP2.finding_the_bubbled_characters(sorted_xy_keypoints_s12_p1,
                                                               alignedCircWithLettList_s12_p1)
            self.characters_s12_p2 = self.engP2.finding_the_bubbled_characters(sorted_xy_keypoints_s12_p2,
                                                               alignedCircWithLettList_s12_p2)

            self.characters_s11_bp_c1 = self.engP2.finding_the_bubbled_characters(sorted_xy_keypoints_s11_bp_s1,
                                                                  alignedCircWithSubjList_s11_bp_s1)
            self.characters_s11_bp_c2 = self.engP2.finding_the_bubbled_characters(sorted_xy_keypoints_s11_bp_s2,
                                                                  alignedCircWithSubjList_s11_bp_s2)
            self.characters_s11_bp_c3 = self.engP2.finding_the_bubbled_characters(sorted_xy_keypoints_s11_bp_s3,
                                                                  alignedCircWithSubjList_s11_bp_s3)
            self.characters_s12_bp_c1 = self.engP2.finding_the_bubbled_characters(sorted_xy_keypoints_s12_bp_s1,
                                                                  alignedCircWithSubjList_s12_bp_s1)
            self.characters_s12_bp_c2 = self.engP2.finding_the_bubbled_characters(sorted_xy_keypoints_s12_bp_s2,
                                                                  alignedCircWithSubjList_s12_bp_s2)
            self.characters_s12_bp_c3 = self.engP2.finding_the_bubbled_characters(sorted_xy_keypoints_s12_bp_s3,
                                                                      alignedCircWithSubjList_s12_bp_s3)
            characters_s11_bp_selection_list = []
            characters_s11_bp_selection_list.append(self.characters_s11_bp_c1)
            characters_s11_bp_selection_list.append(self.characters_s11_bp_c2)
            characters_s11_bp_selection_list.append(self.characters_s11_bp_c3)

            characters_s12_bp_selection_list = []
            characters_s12_bp_selection_list.append(self.characters_s12_bp_c1)
            characters_s12_bp_selection_list.append(self.characters_s12_bp_c2)
            characters_s12_bp_selection_list.append(self.characters_s12_bp_c3)

            for i in characters_s11_bp_selection_list:
                for j in range(len(i)):
                    if i[j] != '':
                        self.characters_11_bp_selection = i[-1]

            for j in characters_s12_bp_selection_list:
                for i in range(len(j)):
                    if j[i] != '':
                        self.characters_12_bp_selection = j[-1]

            self.df3.loc[len(self.df3), ['Тегі-Фамилия', 'Аты-Имя', 'ЖСН-ИИН', 'НҰСҚА-ВАРИАНТ','Сынып литерасы-Литера класса',
                                             'Қосымша сектор-Резервный сектор', 'Математикалық сауаттылық', 'Оқу сауаттылығы', 'Қазақстан тарихы',
                                             'Бейіндік пән 1','Жауап Бөлік 11','Жауап Бөлік 12','Бейіндік пән 2',
                                             'Жауап Бөлік 21', 'Жауап Бөлік 22']] \
                = [self.characters_s1[-1], self.characters_s2[-1], self.characters_s3[-1], self.characters_s4[-1], self.characters_s5[-1],
                   self.characters_s6[-1], self.characters_s8[-1], self.characters_s9[-1], self.characters_s10[-1], self.characters_11_bp_selection,
                   self.characters_s11_p1[-1], self.characters_s11_p2[-1], self.characters_12_bp_selection, self.characters_s12_p1[-1], self.characters_s12_p2[-1]]



            # print(self.characters_s8)
            # print(self.characters_s9)
            # print(self.characters_s10)
            # print(self.characters_s11_p1)
            # print(self.characters_s11_p2)
            # print(self.characters_s12_p1)
            # print(self.characters_s12_p2)
            # print(self.characters_s11_bp_c1)
            # print(self.characters_s11_bp_c2)
            # print(self.characters_s11_bp_c3)
            # print(self.characters_s12_bp_c1)
            # print(self.characters_s12_bp_c2)
            # print(self.characters_s12_bp_c3)
            print("\n")
            print("\n")
            print("\n")


        # self.df.to_excel('grades.xlsx', index=False)
        self.df3.to_excel('applicants_marked_data.xlsx', index=False)

        self.check_with_answers()


    def check_with_answers(self):
        print("checking with answers...")
        print("\n")

        applicants_marked_df = self.df3.replace(r'\s+', np.nan, regex=True)
        answer_keys_df = self.df2

        for i in range(len(applicants_marked_df)):
            print(applicants_marked_df['НҰСҚА-ВАРИАНТ'][i])


        # print(self.df3['НҰСҚА-ВАРИАНТ'])
        # self.df3 = self.df3.replace(r'\s+', np.nan, regex=True)
        #
        # correct = 0
        # uncorrect = 0

        # for j in range(len(self.df2['НҰСҚА-ВАРИАНТ'])):
        #     print("НҰСҚА-ВАРИАНТ:", self.df2['НҰСҚА-ВАРИАНТ'][j])
        #     for i in range(len(self.df3['НҰСҚА-ВАРИАНТ'])):
        #         if math.isnan(float(self.df3['НҰСҚА-ВАРИАНТ'][i])):
        #             # self.df3.loc[i,'НҰСҚА-ВАРИАНТ']='0000'
        #             # self.df3.to_excel('applicants_marked_data.xlsx', index=False)
        #             continue
                # print(self.df3['НҰСҚА-ВАРИАНТ'][i])
                # print(self.df2['НҰСҚА-ВАРИАНТ'][j])
                # if int(self.df2['НҰСҚА-ВАРИАНТ'][j]) == int(self.df3['НҰСҚА-ВАРИАНТ'][i]):
                #     print('Variants are equal: %d - %d' % (int(self.df2['НҰСҚА-ВАРИАНТ'][j]), int(self.df3['НҰСҚА-ВАРИАНТ'][i])))
                #
                    # for ans_index in range(len(self.df2['Математикалық сауаттылық'][j])):
                    #     original_ans = self.df2['Математикалық сауаттылық'][j][ans_index]
                    #     student_ans = self.df3['Математикалық сауаттылық'][i][ans_index]
                    #     if original_ans == student_ans:
                    #         # print("Original answer: %s, Student answer: %s"% (original_ans, student_ans))
                    #         correct += 1
                    #     else:
                    #         uncorrect += 1

                    # print(correct)
                    # print(uncorrect)
                    # score = float((correct / 20.0) * 100)
                    # print(score)
                    # if self.df3['Математикалық сауаттылық'][i] ==
                    #     correct = 0
                    # uncorrect = 0
                    # print(self.df3['Математикалық сауаттылық'][j])
                    # print(self.df2['Математикалық сауаттылық'][i])


                    # for i in range(len(characters_s8) - 1):
                    #     if characters_s8[i] == ANSWER_KEY[i]:
                    #         print(characters_s8[i])
                            # correct += 1
                        # else:
                        #     uncorrect += 1
                    #
                    # print(correct)
                    # print(uncorrect)
                    #
                    # score = float((correct / 20.0) * 100)
                    #
                    # self.df.loc[[len(self.df)], 'Математикалық сауаттылық:'] = score
                    # print(self.df)
                    # print("Length of Data Frame: ", len(self.df))

        # if self.df['НҰСҚА-ВАРИАНТ'] == self.df2['Нұсқа']:
        #     print(self.df['НҰСҚА-ВАРИАНТ'][0], self.df2['Нұсқа'][0])


    @pyqtSlot()
    def loadFileWithAnswers(self):
        filePath = QFileDialog().getOpenFileName()
        if len(filePath) == 2:
            print("The file path: ", filePath[0])
            splited_file_path = filePath[0].split('/')
            file_name = splited_file_path[-1]
            print("The file name: ", file_name)
            splited_file_name = file_name.split('.')
            print(splited_file_name)
            if splited_file_name[-1] != 'xlsx':
                print('Invalid file!')
            else:
                print("File is in .xlsx format!")
                # df = pd.read_excel('file_with_answers.xlsx')
                self.df2 = pd.read_excel(file_name)
                self.loadImageBtn.setEnabled(1)
                self.scanBtn.setEnabled(1)


                # if df2['НҰСҚА-ВАРИАНТ'][0] == df['Нұсқа'][0]:
                #     print("Нұсқа: ", df2['НҰСҚА-ВАРИАНТ'][0])


# if __name__ == '__main__':
app = QApplication(sys.argv)
a_window = ScanningTestApp()
a_window.show()
sys.exit(app.exec_())
