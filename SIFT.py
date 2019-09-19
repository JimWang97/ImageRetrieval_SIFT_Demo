import cv2
import math
import numpy as np
import heapq


class SIFT():
    def __init__(self, img):
        self.init_sigma = 1.6
        self.octvs = 5  # 五层
        self.imgOfLayers = 3  # 最后获得三张特征图 需要每组有6张图
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (0, 0), fx=2, fy=2)  # 扩大一倍的原图作为第一层
        img = cv2.GaussianBlur(img, (3, 3), math.sqrt(self.init_sigma ** 2 - (2 * 0.5) ** 2))
        self.img = img
        self.gauss_pyr = []  # 高斯金字塔 5层 6组
        self.dog_pyr = []  # 差分高斯金字塔 5层 5组
        self.norm_pyr = []  # 归一化的差分金字塔 5层 5组
        self.extr_pyr = []  # 特征点金字塔 5层 3组
        self.mag_pyr = []  # 梯度金字塔 5层 3组
        self.ori_pyr = []  # 方向金字塔 5层 3组
        self.featuresPoint = []  # 特征点数组 内部是字典
        self.featureInfo = []  # 特征点、方向、大小 内部是字典
        self.descriptors = []  # 描述子
        for i in range(self.octvs):
            self.gauss_pyr.append(np.zeros((self.imgOfLayers + 3, img.shape[1] // (2 ** i), img.shape[0] // (2 ** i))))
            self.dog_pyr.append(np.zeros((self.imgOfLayers + 2, img.shape[0] // (2 ** i), img.shape[1] // (2 ** i))))
            self.extr_pyr.append(
                np.zeros((self.imgOfLayers, self.img.shape[1] // (2 ** i), self.img.shape[0] // (2 ** i))))
            # self.norm_pyr.append(
            #     np.zeros((self.imgOfLayers + 2, self.img.shape[1] // (2 ** i), self.img.shape[0] // (2 ** i))))
            self.mag_pyr.append(
                np.zeros((self.imgOfLayers, self.img.shape[1] // (2 ** i), self.img.shape[0] // (2 ** i))))
            self.ori_pyr.append(
                np.zeros((self.imgOfLayers, self.img.shape[1] // (2 ** i), self.img.shape[0] // (2 ** i))))

    def start(self):
        self.constructPyr()
        self.findKeyPoint()
        self.calMagOri()
        self.genDesc()
        return sorted(self.descriptors, key=lambda x: x.__getitem__('scl'))

    def constructPyr(self):
        print("开始构造高斯金字塔...")
        s = self.imgOfLayers
        k = 2 ** (1 / s)

        sigma = []
        sigma.append(self.init_sigma)
        for i in range(1, self.imgOfLayers + 3):
            sigma.append(sigma[i - 1] * k)

        # 构造高斯金字塔
        for i in range(self.octvs):
            for j in range(self.imgOfLayers + 3):
                if i == 0 and j == 0:
                    self.gauss_pyr[i][j, :, :] = self.img
                elif j == 0:
                    self.gauss_pyr[i][j, :, :] = cv2.resize(self.gauss_pyr[i - 1][self.imgOfLayers, :, :], (0, 0),
                                                            fx=0.5, fy=0.5)
                else:
                    self.gauss_pyr[i][j, :, :] = cv2.GaussianBlur(self.gauss_pyr[i][j - 1, :, :], (
                        2 * round(4.0 * sigma[j] + 0.5) + 1, 2 * round(4.0 * sigma[j] + 0.5) + 1), sigma[j])

        # 构造差分高斯金字塔
        print("开始构造差分高斯金字塔...")

        for i in range(self.octvs):
            for j in range(self.imgOfLayers + 2):
                self.dog_pyr[i][j, :, :] = self.gauss_pyr[i][j + 1, :, :] - self.gauss_pyr[i][j, :, :]

    def findKeyPoint(self):
        print("开始寻找极值点...")
        img_border = 5
        maxIter = 5
        contr_thr = 0.04
        curv_thr = 10

        for i in range(self.octvs):
            for j in range(1, self.imgOfLayers + 1):
                prelim_contr_thr = 0.5 * contr_thr / (i + 1)  # 0.5 * T / s 当前层数
                dogimgs = self.dog_pyr[i]
                dogimg = self.dog_pyr[i][j, :, :]
                for x in range(img_border + 1, dogimg.shape[1] - img_border):
                    for y in range(img_border + 1, dogimg.shape[0] - img_border):
                        dict = {}
                        # 设阈值剔除
                        if abs(dogimg[x, y]) < prelim_contr_thr:
                            continue

                        # 求八领域 上下相邻组内图像块极值
                        maxbool = False
                        minbool = False
                        nearVal = []
                        for dx in range(-1, 2):
                            for dy in range(-1, 2):
                                for dj in range(-1, 2):
                                    nearVal.append(dogimgs[j + dj, x + dx, y + dy])

                        if dogimg[x, y] == max(nearVal):
                            maxbool = True
                        elif dogimg[x, y] == min(nearVal):
                            minbool = True

                        # 亚像素求精准极值点
                        if maxbool or minbool:
                            count = 0
                            x_new = x
                            y_new = y
                            intvl = j
                            while count < maxIter:
                                dx = (dogimgs[j, x + 1, y] - dogimgs[j, x - 1, y]) * 0.5 / 255
                                dy = (dogimgs[j, x, y + 1] - dogimgs[j, x, y - 1]) * 0.5 / 255
                                dj = (dogimgs[j + 1, x, y] - dogimgs[j - 1, x, y]) * 0.5 / 255
                                dxx = (dogimgs[j, x + 1, y] + dogimgs[j, x - 1, y] - 2 * dogimgs[j, x, y]) * 1.0 / 255
                                dyy = (dogimgs[j, x, y + 1] + dogimgs[j, x, y - 1] - 2 * dogimgs[j, x, y]) * 1.0 / 255
                                djj = (dogimgs[j + 1, x, y] + dogimgs[j - 1, x, y] - 2 * dogimgs[j, x, y]) * 1.0 / 255
                                dxy = (dogimgs[j, x + 1, y + 1] + dogimgs[j, x - 1, y - 1] - dogimgs[j, x + 1, y - 1] -
                                       dogimgs[j, x - 1, y + 1]) * 0.25 / 255
                                dxj = (dogimgs[j + 1, x + 1, y] + dogimgs[j - 1, x - 1, y] - dogimgs[j - 1, x + 1, y] -
                                       dogimgs[j + 1, x - 1, y]) * 0.25 / 255
                                dyj = (dogimgs[j + 1, x, y + 1] + dogimgs[j - 1, x, y - 1] - dogimgs[j + 1, x, y - 1] -
                                       dogimgs[j - 1, x, y + 1]) * 0.25 / 255

                                dD = np.matrix([[dx], [dy], [dj]])
                                H = np.matrix([[dxx, dxy, dxj], [dxy, dyy, dyj], [dxj, dyj, djj]])

                                # 最小二乘
                                # x^ = - f''/x'' * f'/x'
                                x_hat = np.linalg.lstsq(H, dD)[0]
                                # Dx^ = D + 1/2(D'/x')T * x^
                                Dx_hat = dogimg[x, y] + 0.5 * np.dot(dD.transpose(), x_hat)

                                # Lowe:0.03, Rob Hess:0.04/S
                                if np.absolute(x_hat[0]) < 0.5 and np.absolute(x_hat[1]) < 0.5 and np.absolute(
                                        x_hat[2]) < 0.5 and np.absolute(Dx_hat) > 0.03:
                                    # 消除边缘效应
                                    tr = dxx + dyy
                                    det = dxx * dyy + dxy ** 2
                                    if det > 0 and tr ** 2 / det < (curv_thr ** 2 + 1) / curv_thr:
                                        self.extr_pyr[i][intvl - 1, x_new, y_new] = 1
                                        dict['x'] = x_new
                                        dict['y'] = y_new
                                        dict['x_hat'] = x_hat
                                        dict['octv'] = i
                                        dict['intvl'] = intvl
                                        dict['sigma_octv'] = self.init_sigma * (
                                                (intvl + float(x_hat[2] - 1)) / self.imgOfLayers) ** 2
                                        self.featuresPoint.append(dict)
                                        break

                                x_new = x_new + round(float(x_hat[0]))
                                y_new = y_new + round(float(x_hat[1]))
                                intvl = intvl + round(float(x_hat[2]))
                                count += 1

        print("共找到%d特征点" % (len(self.featuresPoint)))

    def calMagOri(self):
        '''

        :return: 特征点数组 list[dict]
        '''

        print("开始计算特征点梯度与方向")
        bins = 36
        sigma_factor = 1.5
        for item in self.featuresPoint:

            x = item['x']
            y = item['y']
            # 实际上这张特征图来自j+1图 与j j+2计算而来。 因此计算梯度时需要用j+1来计算
            hists = self.oriHist(item['octv'], item['intvl'], x, y, bins, sigma_factor)
            for _ in range(2):  # 做两次平滑
                hists = sum(self.smoothHists(hists, bins).tolist(), [])

            mainOir_mag = max(hists)
            # 寻找辅助方向
            for num in range(bins):
                dict = {}
                ori_mag = heapq.nlargest(num + 1, hists)[0]
                ori_mag_index = hists.index(ori_mag)
                if ori_mag_index >= 0.8 * mainOir_mag:
                    if ori_mag_index == 1:
                        l = bins - 1
                        r = ori_mag_index + 1
                    elif ori_mag_index == bins - 1:
                        l = ori_mag_index - 1
                        r = 0
                    else:
                        l = ori_mag_index - 1
                        r = ori_mag_index + 1
                    bin = abs(ori_mag_index + self.interp_hist(hists[l], ori_mag, hists[r])) % bins
                    accu_intvl = item['intvl'] + item['x_hat'][2]
                    # 缩放到原始尺寸
                    dict['x'] = (x + item['x_hat'][0]) * 2 ** (item['octv'] - 2)
                    dict['y'] = (y + item['x_hat'][1]) * 2 ** (item['octv'] - 2)
                    dict['scl'] = self.init_sigma * (item['octv'] - 2 + (accu_intvl - 1) / self.imgOfLayers) ** 2
                    dict['ori'] = bin / bins * 2 * math.pi - math.pi
                    dict['feat_ind'] = self.featuresPoint.index(item)
                    self.featureInfo.append(dict)

    def oriHist(self, oct, imgLayer, x, y, bins, sigma_factor):
        hists = np.zeros((bins, 1))
        ori_sigma = sigma_factor * (oct + 1)
        r = round(3 * ori_sigma)
        img = self.gauss_pyr[oct][imgLayer, :, :]
        exp_factor = 2 * ori_sigma * ori_sigma  # 计算高斯分布

        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                mag, ori = self.calGrad(img, x + dx, y + dy)
                if mag != -1 and ori != -1:
                    w = math.exp(-(dx ** 2 + dy ** 2) / exp_factor)
                    bin = (round(bins * (ori + math.pi) / (2 * math.pi))) % bins
                    hists[bin] = hists[bin] + w * mag

        return hists

    def calGrad(self, img, x, y):
        width = img.shape[1]
        height = img.shape[0]
        mag = -1
        ori = -1
        if x > 0 and x < width - 1 and y > 0 and y < height - 1:
            dx = img[x - 1, y] - img[x + 1, y]
            dy = img[x, y + 1] - img[x, y - 1]
            mag = math.sqrt(dx * dx + dy * dy)
            ori = math.atan(dx * 1.0 / dy)

        return mag, ori

    def smoothHists(self, hists, bins):
        hists_new = np.zeros((bins, 1))
        for i in range(bins):
            if i == 0:
                prev = hists[bins - 1]
                next = hists[i + 1]
            elif i == bins - 1:
                prev = hists[i - 1]
                next = hists[0]
            else:
                prev = hists[i - 1]
                next = hists[i + 1]
            hists_new[i] = 0.25 * prev + 0.5 * hists[i] + 0.25 * next

        return hists_new

    def interp_hist(self, l, c, r):
        '''
        直方图的抛物插值，确定准确定位方向
        :param l: 前一个直方图方向
        :param c: 当前直方图方向
        :param r: 后一个直方图方向
        :return: 插值结果方向
        '''
        return 0.5 * (l - r) / (l - 2 * c + r)

    def genDesc(self):
        '''
        生成描述子
        :return: 4*4*8 共128维特征向量
        '''
        patches = 4
        oriBins = 8  # 取八个方向的直方图
        mag_thr = 0.2  # 过大梯度影响归一化
        desc_length = patches * patches * oriBins
        for feat in self.featureInfo:
            feature_point = self.featuresPoint[feat['feat_ind']]
            gauss_img = self.gauss_pyr[feature_point['octv']][feature_point['intvl'], :, :]
            hist_width = 3 * feature_point['sigma_octv']
            radius = round(hist_width * (patches + 1) * math.sqrt(2) / 2)  # 考虑旋转
            x = feature_point['x']
            y = feature_point['y']
            feat_ori = feat['ori']
            hist = np.zeros((desc_length,))
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    j_rota = j * math.cos(feat_ori) - i * math.sin(feat_ori)
                    i_rota = j * math.sin(feat_ori) - j * math.cos(feat_ori)
                    r_bin = i_rota / hist_width + patches / 2 - 0.5
                    c_bin = j_rota / hist_width + patches / 2 - 0 / 5
                    if r_bin > -1 and r_bin < patches and c_bin > -1 and c_bin < patches:
                        mag, ori = self.calGrad(gauss_img, x + i, y + j)
                        if mag != -1 and ori != -1:
                            ori = ori - feat_ori
                            while ori < 0:
                                ori = ori + 2 * math.pi

                            while ori > 2 * math.pi:
                                ori = ori - 2 * math.pi

                            o_bin = ori * oriBins / (2 * math.pi)
                            w = math.exp(-(j_rota * j_rota + i_rota * i_rota) / (2 * (0.5 * patches * hist_width) ** 2))
                            hist = self.interpHistGrad(hist, r_bin, c_bin, o_bin, mag * w, patches, oriBins)
            self.descriptors.append(self.hist2Desc(feat, hist, mag_thr))

    def interpHistGrad(self, hist, r, c, o, mag, d, obins):
        '''
        三线性插值
        :param hist:
        :param r:
        :param c:
        :param o:
        :param mag:
        :param d:
        :param obins:
        :return:
        '''
        r0 = math.floor(r)
        c0 = math.floor(c)
        o0 = math.floor(o)
        d_r = r - r0
        d_c = c - c0
        d_o = o - o0

        for i in range(2):
            r_index = r0 + i
            if r_index >= 0 and r_index < d:
                for j in range(2):
                    c_index = c0 + j
                    if c_index >= 0 and c_index < d:
                        for k in range(2):
                            o_index = (o0 + k) % obins
                            value = mag * (0.5 + (d_r - 0.5) * (2 * i - 1)) * (0.5 + (d_c - 0.5) * (2 * j - 1)) * (
                                    0.5 + (d_o - 0.5) * (2 * k - 1))
                            hist_index = r_index * d * obins + c_index * obins + o_index
                            hist[hist_index] = hist[hist_index] + value

        return hist

    def hist2Desc(self, feat, hist, mag_thr):
        '''

        :param feat:
        :param hist:
        :param mag_thr:
        :return:
        '''
        desc = {}
        hist = hist / np.max(hist)
        for i in range(len(hist)):
            hist[i] = min(hist[i], mag_thr)
        hist = hist / np.max(hist)
        desc['desc'] = hist
        desc['x'] = feat['x'].A[0][0]
        desc['y'] = feat['y'].A[0][0]
        desc['scl'] = feat['scl'].A[0][0]
        return desc
