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
        print("原图像特征计算...")
        featA = self.getFeat(queryImg)

        similarity = {}
        for imgName in dataset:
            img = cv2.imread(imgName)
            print("目标图像(%s)特征计算..." %imgName)
            featB = self.getFeat(img)
            similarity[imgName] = self.CosSim(featA, featB)

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

    def getFeat(self, img):
        sift = SIFT(img)
        siftDescr = sift.start()
        siftVec = []
        for item in siftDescr:
            siftVec.append(item['desc'])
        siftVec = np.array(siftVec)
        return siftVec

    def CosSim(self, siftVecA, siftVecB):
        distRatio = 0.6
        count = 0

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