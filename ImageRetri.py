import glob
import cv2
import os
from SIFT import SIFT
import numpy as np
import heapq


class ImageRetri():
    def __init__(self, datasetRoot, queryImgName, returnImgNum):
        self.datasetRoot = datasetRoot
        self.queryImgName = queryImgName
        self.returnImgNum = returnImgNum
        self.Retri()

    def Retri(self):
        queryImg = cv2.imread(self.queryImgName)
        queryImgPath = '/'.join(self.queryImgName.split('/')[:-1])

        RealTrueImgName = glob.glob(pathname=queryImgPath + '/*.jpg')
        RealTrueNum = len(RealTrueImgName)

        dataset = []
        for root, dirs, files in os.walk(self.datasetRoot, True):
            for file in files:
                if file[-3:] == 'jpg':
                    dataset.append(root + '/' + file)

        similarity = {}
        for imgName in dataset:
            img = cv2.imread(imgName)
            similarity[imgName] = self.CosSim(queryImg, img)

        res = sorted(similarity.items(), key=lambda x: x[1], reverse=True)
        returnImg = res[:self.returnImgNum]
        returnTrueNum = 0
        returnFalseNum = 0
        for item in returnImg:
            if item.key() in RealTrueImgName:
                returnTrueNum += 1
            else:
                returnFalseNum += 1
        recall = returnTrueNum / RealTrueNum
        precision = returnTrueNum / self.returnImgNum
        print("recall = %d, precision = %d." % (recall, precision))

    def CosSim(self, queryImg, img):
        distRatio = 0.6
        count = 0
        siftA = SIFT(queryImg)
        siftB = SIFT(img)
        print("获取原图像特征...")
        siftDescrA = siftA.start()  # 描述子,x,y,尺度
        print("原图像特征获取完毕...")
        print("获取目标图像特征...")
        siftDescrB = siftB.start()  # 描述子,x,y,尺度
        print("目标图像特征获取完毕...")
        siftVecA = []
        siftVecB = []
        for item in siftDescrA:
            siftVecA.append(item['desc'])
        for item in siftDescrB:
            siftVecB.append(item['desc'])

        siftVecA = np.array(siftVecA)
        siftVecB = np.array(siftVecB)

        match = np.zeros((len(siftVecA)))
        siftVecB_T = np.transpose(siftVecB)
        val = np.matmul(siftVecA, siftVecB_T)  # 图A描述子个数 × 图B描述子个数
        val = np.arccos(val)
        for i in range(len(match)):
            vec = val[i, :]
            vec = vec[~np.isnan(vec)]
            small1 = heapq.nsmallest(1, vec)[0]
            small2 = heapq.nsmallest(2, vec)[0]
            if small1 < distRatio * small2:
                count = count + 1
                match[i] = 1

        if count >= len(match)/2:
            print("匹配成功")

        return count