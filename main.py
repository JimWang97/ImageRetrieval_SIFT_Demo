# -*- coding: utf-8 -*-
import cv2
import sys
import string
import glob
from ImageRetri import ImageRetri


def main():
    '''

    :return:
    '''
    returnImgNum = 5
    queryImgName = r"./image/A0C632/A0C632_20151103070227_6595232074.jpg"
    root = r"./image"
    ImageRetri(root, queryImgName, returnImgNum)


if __name__ == '__main__':
    main()