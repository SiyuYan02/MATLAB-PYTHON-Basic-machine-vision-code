import numpy as np
import cv2
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import selectivesearch
import matplotlib.pyplot as plt


def ShowImg_plt(title, img):
    tmp = img[:, :, [2, 1, 0]]
    plt.figure()
    plt.imshow(tmp)
    plt.title(title)
    plt.show()


def predict(model, target_size, top_n, sub_img):

    # 输入：
    # model：分类模型  model = ResNet50(weights='imagenet')
    # target_size：输入图像需要resize的大小 target_size = (224, 224)
    # top_n:输出概率最高的几个类 top_n=1
    # img：待预测图像 img = cv2.imread("1.jpg")
    # 输出：
    # 图像预测结果的元组列表（类、描述、概率）
    # 调整数组形状：

    # 调整sub_img的形状
    if sub_img.shape[0] != target_size[0] or sub_img.shape[1] != target_size[1]:
        sub_img = cv2.resize(sub_img, target_size)

    sub_img = np.expand_dims(sub_img, axis=0)
    sub_img = preprocess_input(sub_img)

    # 预测
    pred = model.predict(sub_img)
    return decode_predictions(pred, top=top_n)[0]


def GetRect(img):

    #输入:
    #img: 输入待处理图像图像
    #输出
    #经过selectivesearch检测并进行删减之后的矩形框

    # 用selectivesearch进行候选区选择
    # 选择性搜索
    img_lbl, rects = selectivesearch.selective_search(img, scale=400, sigma=0.9, min_size=10)

    # 前处理
    height, width = img.shape[0], img.shape[1]

    candidates = set()
    for r in rects:
        # 排除大小不合适的矩形框
        x, y, w, h = r['rect']
        if r['size'] < 2000:
            continue
        if w < width*  0.1 or w > width * 0.5:
            continue
        if h < height * 0.1 or h > height * 0.3:
            continue
        # 排除重复的矩形框
        if r['rect'] in candidates:
            continue
        # 排除长宽比不合适的矩形
        if w / h > 5 or h / w > 5:
            continue
            # 排除小于 2000 pixels的候选区域(并不是bounding box中的区域大小)



        candidates.add(r['rect'])
    return candidates


def None_Max_Supression(bounding_boxes, confidence_score, classifications, IOU_thred):
        # 输入：
    # bounding_boxes: 矩形框的尺寸
    # confidence_score: 设定的识别置信度阈值
    # classifications: 对应预测出的阈值
    # IOU_thred: 函数中设定好的IOU置信度阈值
    # 输出：
    # 选中的矩形框的大小、置信度、种类

    # 将list转为array
    boxes = np.array(bounding_boxes)
    score = np.array(confidence_score)

    # 获得四个坐标点分别对应的array
    start_x = boxes[:, 0]
    start_y = boxes[:, 1]
    end_x = boxes[:, 2]
    end_y = boxes[:, 3]

    # 最终返回的经过非极大值抑制之后的最终筛选结果
    picked_boxes = []
    picked_score = []
    picked_class = []

    # 计算每一个矩形框的面积
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    # 选择当前候选矩形框中具有最大置信度的矩形框
    order = np.argsort(score)

    # 开始迭代筛选
    while order.size > 0:
        index = order[-1]

        # 把最大置信度的矩形放到最终的结果中
        picked_boxes.append(bounding_boxes[index])
        picked_score.append(confidence_score[index])
        picked_class.append(classifications[index])

        # 计算每一个矩形对应于最大置信度矩形的重叠区域
        x1 = np.maximum(start_x[index], start_x[order[:-1]])
        x2 = np.minimum(end_x[index], end_x[order[:-1]])
        y1 = np.maximum(start_y[index], start_y[order[:-1]])
        y2 = np.minimum(end_y[index], end_y[order[:-1]])

        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h

        # 计算相应的IOU
        ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)

        # 删除小于设定IOU的矩形
        left = np.where(ratio < IOU_thred)
        order = order[left]

    return picked_boxes, picked_score, picked_class


def GetClassification(img, candidates, special_class=None, confidence_thred=0.01, IOU_thred=0.3):
    # 输入：
    # IOU_thred: 进行非极大抑制时的阈值，目的是去除重合面积较大的矩形框
    # confidence: 识别置信度阈值
    # special_class: 需要检测出的特定类别
    # img: 输入待处理的图像
    # candidates: 经过选择性搜索以及前处理获得的候选矩形框
    # 输出：
    # 每一个矩形的大小、对应概率以及分类

    bounding_boxes = []  # 选中的矩形框的尺寸
    confidence_score = []  # 对应预测值的置信度
    classifications = []  # 类别所构成的list

    img = img.copy()
    model = ResNet50(weights='imagenet')  # 载入ResNet50网络模型，并使用在ImageNet ILSVRC比赛中已经训练好的权重
    target_size = (224, 224)  # ResNet50的输入大小固定为(224,224)，其他大小会报错
    top_n = 1  # 只输出最高概率对应的一类

    for x, y, w, h in candidates:
        sub_img = img[y:y + h, x:x + w, :]  # 获得矩形框的图像
        pred = predict(model, target_size, top_n, sub_img)  # 预测sub-img所属类别

        '''删除不符合所指定类别的矩形框'''
        classification = pred[0][1]
        # if special_class is not None and special_class != classification:
        #     continue
        if 'desktop_computer' == classification or 'screen' == classification  \
        or 'monitor' == classification  or 'laptop'== classification :
        #去除掉置信度过低的矩形框
            if pred[0][2] > confidence_thred:
                bounding_boxes.append([x, y, x + w, y + h])
                confidence_score.append(pred[0][2])
                classifications.append(classification)

    '''后处理——非极大值抑制'''
    picked_boxes, picked_score, picked_class = None_Max_Supression(bounding_boxes, confidence_score,
                                                                   classifications, IOU_thred)
    for idx in range(len(picked_boxes)):
        x1, y1, x2, y2 = picked_boxes[idx]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_AA)
        img = cv2.putText(img, picked_class[idx], (x1 - 10, y1), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

    return img


if __name__ == '__main__':
    img = cv2.imread("result_37.jpg")
    candidates = GetRect(img)  # 获取候选矩形框
    res_img = GetClassification(img, candidates)  # 返回框选后的结果图像
    ShowImg_plt('$Object\quad Detection$', res_img)
    cv2.imwrite('result_45.jpg', res_img)