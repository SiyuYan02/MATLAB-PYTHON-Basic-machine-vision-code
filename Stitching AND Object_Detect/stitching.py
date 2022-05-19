import cv2
import numpy as np
import math
import os
import matplotlib.pyplot as plt


def detectAndDescribe(image):
    # 将彩色图片转换成灰度图
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 建立SIFT算子
    sift = cv2.xfeatures2d.SIFT_create()

    # 计算出图像的关键点和sift特征向量，kpA表示图像A关键点的坐标,features表示特征向量
    (kps, features) = sift.detectAndCompute(image, None)

    # 将结果转换成NumPy数组
    kps = np.float32([kp.pt for kp in kps])

    # 返回特征点集，及对应的描述特征
    return (kps, features)


def matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
    # 建立暴力匹配器，使用默认的欧氏距离
    matcher = cv2.BFMatcher()

    # 使用KNN检测来自A、B图的SIFT特征匹配对，K=2
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)

    matches = []
    for m in rawMatches:
        # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            # 存储两个点在featuresA, featuresB中的索引值
            matches.append((m[0].trainIdx, m[0].queryIdx))

    # 当筛选后的匹配对大于4时，计算视角变换矩阵
    if len(matches) > 4:
        # 获取匹配对的点坐标
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])

        # 计算视角变换矩阵
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

        # 返回结果
        return (matches, H, status)

    # 如果匹配对小于4时，返回None
    return None


def drawMatches(imageA, imageB, kpsA, kpsB, matches, status):
    # 初始化可视化图片，将A、B图左右连接到一起
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB

    # 联合遍历，画出匹配对
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        # 当点对匹配成功时，画到可视化图上
        if s == 1:
            # 画出匹配对
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

    # 返回可视化结果
    return vis


def stitch(images, ratio=0.75, reprojThresh=4.0,showMatches=False):
        #获取输入图片
        (imageB, imageA) = images
        #检测A、B图片的SIFT关键特征点，并计算特征描述子
        (kpsA, featuresA) = detectAndDescribe(imageA)
        (kpsB, featuresB) = detectAndDescribe(imageB)

        # 匹配两张图片的所有特征点，返回匹配结果
        M = matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        # 如果返回结果为空，没有匹配成功的特征点，退出算法
        if M is None:
            return None

        # 否则，提取匹配结果
        # H是3x3视角变换矩阵
        (matches, H, status) = M

        # 将图片A进行视角变换，result是变换后图片
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], max(imageA.shape[0],imageB.shape[0])))

        row_B, col_B, dB = imageB.shape
        for i in range(row_B):
            for j in range(col_B):
                if result[i][j][0] == 0 and result[i][j][1] == 0 and result[i][j][2] == 0:
                    result[i][j] = imageB[i][j]
                else:
                    x = j / col_B
                    a=x**9  # A的比例
                    b = 1 - a  # B的比例
                    result[i][j] = a * result[i][j] + b * imageB[i][j]
        # 将图片B传入result图片最左端

        if showMatches:
            # 生成匹配图片
            vis = drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            # 返回结果
            return (result, vis)

        global edges_x
        # global edges_x2
        #去黑边
        row_R, col_R, dR = result.shape
        for j in range(col_R):
            if result[0][j][0] == 0 and result[0][j][1] == 0 and result[0][j][2] == 0:
                edges_x=j
                break
        # for j in range(col_R):
        #     if result[row_R - 1][j][0] == 0 and result[row_R - 1][j][1] == 0 and result[row_R - 1][j][2] == 0:
        #         edges_x2 = j
        #         break
        # edges_x = min(edges_xx, edges_x2)
        # gray2 = result[:, :edges_x-1]
        return result

        # result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
        # 检测是否需要显示图片匹配

#柱形变换
def cylindrical_projection(img, f):
    rows = img.shape[0]
    cols = img.shape[1]

    # f = cols / (2 * math.tan(np.pi / 8))

    blank = np.zeros_like(img)
    center_x = int(cols / 2)
    center_y = int(rows / 2)

    for y in range(rows):
        for x in range(cols):
            theta = math.atan((x - center_x) / f)
            point_x = int(f * math.tan((x - center_x) / f) + center_x)
            point_y = int((y - center_y) / math.cos(theta) + center_y)

            if point_x >= cols or point_x < 0 or point_y >= rows or point_y < 0:
                pass
            else:
                blank[y, x, :] = img[point_y, point_x, :]
    return blank

#去黑边
def cutblackedges(result):
    row_R, col_R, dR = result.shape
    global edges_xmin, edges_xmax
    for j in range(col_R):
        if result[int(row_R/2)][j][0] != 0 and result[int(row_R/2)][j][1] != 0 and result[int(row_R/2)][j][2] != 0:
            edges_xmin = j
            break
    for j in reversed(range(col_R)):
        if result[int(row_R/2)][j][0] != 0 and result[int(row_R/2)][j][1] != 0 and result[int(row_R/2)][j][2] != 0:
            edges_xmax = j
            break
    row_R, col_R, dR = result.shape
    for j in range(col_R):
        if result[0][j][0] == 0 and result[0][j][1] == 0 and result[0][j][2] == 0:
            edges_x = j
            break
    # for j in range(col_R):
    #     if result[row_R - 1][j][0] == 0 and result[row_R - 1][j][1] == 0 and result[row_R - 1][j][2] == 0:
    #         edges_x2 = j
    #         break
    # edges_x = min(edges_xx, edges_x2)
    # gray2 = result[:, :edges_x-1]
    res = result[:, :edges_xmax - 1]
    return res


def pltshowim(title, img):
    tmp = img[:, :, [2, 1, 0]]
    plt.figure()
    plt.imshow(tmp)
    plt.title(title)
    plt.show()


img_1 = cv2.imread("result_40.png")
# img_1_=cylindrical_projection(img_1,1000)
# img_1_=cutblackedges(img_1_)

img_2 = cv2.imread("result_39.png")
# img_2_=cylindrical_projection(img_2,500)
# img_2_=cutblackedges(img_2)

result = stitch(images = [img_1,img_2], showMatches=False)
result=cutblackedges(result)
cv2.imwrite('result_42.png',result)
# result=cutblackedges(result)
pltshowim('result',result)




