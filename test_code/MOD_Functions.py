import cv2
import numpy as np
from torch import nn
from torchvision import transforms
import torch.nn.functional as F
import torch
import xml.etree.ElementTree as ET
import onnxruntime as ort
from PIL import Image

from torchvision.models import resnet18


def ECC_stablize(frame1, frame2):
    # grid-based KLT tracking
    blur_kernel = 11
    prevFrame = cv2.GaussianBlur(frame1, (blur_kernel, blur_kernel), 0)  # 高斯模糊，用于去噪
    prevFrame = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)  # 灰度化
    img1 = prevFrame

    currentFrame = cv2.GaussianBlur(frame2, (blur_kernel, blur_kernel), 0)
    currentFrame = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2GRAY)
    img2 = currentFrame

    sz = img1.shape

    warp_mode = cv2.MOTION_AFFINE

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    number_of_iterations = 100
    termination_eps = 1e-3
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TermCriteria_COUNT, number_of_iterations, termination_eps)

    cc, warp_matrix = cv2.findTransformECC(img1, img2, warp_matrix, warp_mode, criteria, None, 11)
    dx = warp_matrix[0, 2]
    dy = warp_matrix[1, 2]

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        img2_aligned = cv2.warpPerspective(img2, warp_matrix, (sz[1], sz[0]),
                                           flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        img_aligned = cv2.warpAffine(img2, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    return int(-dx), int(-dy)


def affine_stablize(frame1, frame2):
    # grid-based KLT tracking
    blur_kernel = 11
    prevFrame = cv2.GaussianBlur(frame1, (blur_kernel, blur_kernel), 0)  # 高斯模糊，用于去噪
    prevFrame = cv2.cvtColor(prevFrame, cv2.COLOR_BGR2GRAY)  # 灰度化

    currentFame = cv2.GaussianBlur(frame2, (blur_kernel, blur_kernel), 0)
    currentFrame = cv2.cvtColor(currentFame, cv2.COLOR_BGR2GRAY)

    lk_params = dict(winSize=(15, 15), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    width = frame2.shape[1]
    height = frame2.shape[0]
    gridSizeW = 32 * 2
    gridSizeH = 24 * 2
    p1 = []
    grid_numW = int(width / gridSizeW - 1)
    grid_numH = int(height / gridSizeH - 1)
    for i in range(grid_numW):
        for j in range(grid_numH):
            point = (np.float32(i * gridSizeW + gridSizeW / 2.0), np.float32(j * gridSizeH + gridSizeH / 2.0))
            p1.append(point)

    p1 = np.array(p1)
    pts_num = grid_numW * grid_numH
    pts_prev = p1.reshape(pts_num, 1, 2)

    pts_cur, st, err = cv2.calcOpticalFlowPyrLK(prevFrame, currentFrame, pts_prev, None, **lk_params)

    # 选择good points
    good_new = pts_cur[st == 1]  # 当前帧中的跟踪点
    good_old = pts_prev[st == 1]  # 前一帧中的跟踪点

    points_new = []
    points_old = []
    # 绘制跟踪框
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        motion_distance0 = np.sqrt((a - c) * (a - c) + (b - d) * (b - d))
        if motion_distance0 > 50:
            continue

        point_new = np.array([a, b])
        point_old = np.array([c, d])
        points_new.append(point_new)
        points_old.append(point_old)

    points_new = np.array(points_new)
    points_old = np.array(points_old)

    # 根据透视变换矩阵计算变换之后的图像
    # homography_matrix, status = cv2.findHomography(points_new, points_old, cv2.RANSAC, 3.0)
    # img_compensate = cv2.warpPerspective(frame2, homography_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    # homo_inv = np.linalg.inv(homography_matrix)

    # 使用仿射变换矩阵进行图像稳像
    # Find affine transformation matrix
    m, _ = cv2.estimateAffinePartial2D(points_new, points_old, maxIters=200, ransacReprojThreshold=3)

    # Extract translation
    dx = m[0, 2]
    dy = m[1, 2]

    # Extract rotation angle
    da = np.arctan2(m[1, 0], m[0, 0])

    # Store transformation
    m = np.zeros((2, 3), np.float32)
    m[0, 0] = np.cos(da)
    m[0, 1] = -np.sin(da)
    m[1, 0] = np.sin(da)
    m[1, 1] = np.cos(da)
    m[0, 2] = dx
    m[1, 2] = dy

    # 根据变换矩阵计算变换之后的图像
    img_compensate = cv2.warpAffine(frame2, m, (width, height))
    m_inv = cv2.invertAffineTransform(m)
    # m_inv = np.linalg.inv(m)

    return int(-dx), int(-dy)


def translate_compensate(frame1, frame2):
    frame1 = cv2.GaussianBlur(frame1, (11, 11), 0)
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    frame2 = cv2.GaussianBlur(frame2, (11, 11), 0)
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # grid-based KLT tracking
    lk_params = dict(winSize=(15, 15), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.003))

    # 创建随机生成的颜色
    # color = np.random.randint(0, 255, (3000, 3))
    # width = frame2.shape[1]
    # height = frame2.shape[0]
    scale = 2

    frame1_grid = cv2.resize(frame1, (960 * scale, 540 * scale), dst=None, interpolation=cv2.INTER_CUBIC)
    frame2_grid = cv2.resize(frame2, (960 * scale, 540 * scale), dst=None, interpolation=cv2.INTER_CUBIC)

    width_grid = frame2_grid.shape[1]
    height_grid = frame2_grid.shape[0]
    gridSizeW = 32 * 2
    gridSizeH = 24 * 2
    p1 = []
    grid_numW = int(width_grid / gridSizeW - 1)
    grid_numH = int(height_grid / gridSizeH - 1)
    for i in range(grid_numW):
        for j in range(grid_numH):
            point = (np.float32(i * gridSizeW + gridSizeW / 2.0), np.float32(j * gridSizeH + gridSizeH / 2.0))
            p1.append(point)

    p1 = np.array(p1)
    pts_num = grid_numW * grid_numH
    pts_prev = p1.reshape(pts_num, 1, 2)

    pts_cur, st, err = cv2.calcOpticalFlowPyrLK(frame1_grid, frame2_grid, pts_prev, None, **lk_params)

    # 选择good points
    good_new = pts_cur[st == 1]  # 当前帧中的跟踪点
    good_old = pts_prev[st == 1]  # 前一帧中的跟踪点

    # points_new = []
    # points_old = []

    # 绘制跟踪框
    # mask0 = np.zeros_like(frame2)  # 为绘制创建掩码图片
    motion_distance = []
    translate_x = []
    translate_y = []
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        motion_distance0 = np.sqrt((a - c) * (a - c) + (b - d) * (b - d))

        if motion_distance0 > 50:
            continue

        translate_x0 = a - c
        translate_y0 = b - d

        # point_new = np.array([a, b])
        # point_old = np.array([c, d])
        # points_new.append(point_new)
        # points_old.append(point_old)

        motion_distance.append(motion_distance0)
        translate_x.append(translate_x0)
        translate_y.append(translate_y0)
        # mask0 = cv2.line(mask0, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 3)
        # cv2.circle(frame2, (int(a), int(b)), 3, color[i].tolist(), -1)

    motion_dst = np.mean(np.array(motion_distance))
    motion_x = np.mean(np.array(translate_x))
    motion_y = np.mean(np.array(translate_y))

    return motion_x, motion_y


def GFTT_compensate(frame1, frame2):
    width = frame2.shape[1]
    height = frame2.shape[0]

    # Good Features To Track
    feature_params = dict(maxCorners=200, qualityLevel=0.3, minDistance=5, blockSize=11)
    # 光流法参数
    # maxLevel 未使用的图像金字塔层数
    lk_params = dict(winSize=(15, 15), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.03))

    # good feature points based method
    pts_prev = cv2.goodFeaturesToTrack(frame1, mask=None, **feature_params)
    pts_cur, st, err = cv2.calcOpticalFlowPyrLK(frame1, frame2, pts_prev, None, **lk_params)
    pts_prev0, st0, err0 = cv2.calcOpticalFlowPyrLK(frame2, frame1, pts_cur, None, **lk_params)
    distance = abs(pts_prev - pts_prev0).reshape(-1, 2).max(-1)
    good = distance < 0.01
    good_new = pts_cur[good == True]
    good_old = pts_prev[good == True]

    homography_matrix, status = cv2.findHomography(good_new, good_old, cv2.RANSAC, 10.0)

    # 根据变换矩阵计算变换之后的图像
    compensated = cv2.warpPerspective(frame1, homography_matrix, (width, height),
                                      flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    # 计算掩膜
    vertex = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32).reshape(-1, 1, 2)
    homo_inv = np.linalg.inv(homography_matrix)
    vertex_trans = cv2.perspectiveTransform(vertex, homo_inv)
    vertex_transformed = np.array(vertex_trans, dtype=np.int32).reshape(1, 4, 2)
    im = np.zeros(frame1.shape[:2], dtype='uint8')
    cv2.polylines(im, vertex_transformed, 1, 255)
    cv2.fillPoly(im, vertex_transformed, 255)
    mask = 255 - im

    return compensated, mask


def motion_compensate(frame1, frame2):
    # grid-based KLT tracking
    lk_params = dict(winSize=(15, 15), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.003))

    # 创建随机生成的颜色
    # color = np.random.randint(0, 255, (3000, 3))
    width = frame2.shape[1]
    height = frame2.shape[0]
    scale = 2

    frame1_grid = cv2.resize(frame1, (960 * scale, 540 * scale), dst=None, interpolation=cv2.INTER_CUBIC)
    frame2_grid = cv2.resize(frame2, (960 * scale, 540 * scale), dst=None, interpolation=cv2.INTER_CUBIC)

    width_grid = frame2_grid.shape[1]
    height_grid = frame2_grid.shape[0]
    gridSizeW = 32 * 2
    gridSizeH = 24 * 2
    p1 = []
    grid_numW = int(width_grid / gridSizeW - 1)
    grid_numH = int(height_grid / gridSizeH - 1)
    for i in range(grid_numW):
        for j in range(grid_numH):
            point = (np.float32(i * gridSizeW + gridSizeW / 2.0), np.float32(j * gridSizeH + gridSizeH / 2.0))
            p1.append(point)

    p1 = np.array(p1)
    pts_num = grid_numW * grid_numH
    pts_prev = p1.reshape(pts_num, 1, 2)

    pts_cur, st, err = cv2.calcOpticalFlowPyrLK(frame1_grid, frame2_grid, pts_prev, None, **lk_params)

    # 选择good points
    good_new = pts_cur[st == 1]  # 当前帧中的跟踪点
    good_old = pts_prev[st == 1]  # 前一帧中的跟踪点

    # points_new = []
    # points_old = []

    # 绘制跟踪框
    # mask0 = np.zeros_like(frame2)  # 为绘制创建掩码图片
    motion_distance = []
    translate_x = []
    translate_y = []
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        motion_distance0 = np.sqrt((a - c) * (a - c) + (b - d) * (b - d))

        if motion_distance0 > 50:
            continue

        translate_x0 = a - c
        translate_y0 = b - d

        # point_new = np.array([a, b])
        # point_old = np.array([c, d])
        # points_new.append(point_new)
        # points_old.append(point_old)

        motion_distance.append(motion_distance0)
        translate_x.append(translate_x0)
        translate_y.append(translate_y0)
        # mask0 = cv2.line(mask0, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 3)
        # cv2.circle(frame2, (int(a), int(b)), 3, color[i].tolist(), -1)
    motion_dist = np.array(motion_distance)
    motion_x = np.mean(np.array(translate_x))
    motion_y = np.mean(np.array(translate_y))

    avg_dst = np.mean(motion_dist)

    # points_new = np.array(points_new)
    # points_old = np.array(points_old)
    # img_optflow = cv2.add(frame2, mask0)
    # cv2.imwrite('./output/drone1_grid/frame_' + str(frameCount) + '.jpg', img_optflow)
    # cv2.imshow('frame with optical flow ', img_optflow)

    # homography_matrix, status = cv2.findHomography(good_new, good_old, cv2.RANSAC, 3.0)
    # homography_matrix, status = cv2.findHomography(points_new, points_old, cv2.RANSAC, 3.0)
    # matchesMask = status.ravel().tolist()
    if len(good_old) < 15:
        homography_matrix = np.array([[0.999, 0, 0], [0, 0.999, 0], [0, 0, 1]])
    else:
        homography_matrix, status = cv2.findHomography(good_new, good_old, cv2.RANSAC, 3.0)

    # 根据变换矩阵计算变换之后的图像
    compensated = cv2.warpPerspective(frame1, homography_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    # 计算掩膜
    vertex = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32).reshape(-1, 1, 2)
    homo_inv = np.linalg.inv(homography_matrix)
    vertex_trans = cv2.perspectiveTransform(vertex, homo_inv)
    vertex_transformed = np.array(vertex_trans, dtype=np.int32).reshape(1, 4, 2)
    im = np.zeros(frame1.shape[:2], dtype='uint8')
    cv2.polylines(im, vertex_transformed, 1, 255)
    cv2.fillPoly(im, vertex_transformed, 255)
    mask = 255 - im

    return compensated, mask, avg_dst, motion_x, motion_y, homography_matrix


def motion_compensate_local(frame1, frame2):
    # grid-based KLT tracking
    lk_params = dict(winSize=(15, 15), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.003))

    # 创建随机生成的颜色
    # color = np.random.randint(0, 255, (3000, 3))

    width = frame2.shape[1]
    height = frame2.shape[0]
    gridSizeW = 8 * 2
    gridSizeH = 8 * 2
    p1 = []
    grid_numW = int(width / gridSizeW - 1)
    grid_numH = int(height / gridSizeH - 1)
    for i in range(grid_numW):
        for j in range(grid_numH):
            point = (np.float32(i * gridSizeW + gridSizeW / 2.0), np.float32(j * gridSizeH + gridSizeH / 2.0))
            p1.append(point)

    p1 = np.array(p1)
    pts_num = grid_numW * grid_numH
    pts_prev = p1.reshape(pts_num, 1, 2)

    pts_cur, st, err = cv2.calcOpticalFlowPyrLK(frame1, frame2, pts_prev, None, **lk_params)

    # 选择good points
    good_new = pts_cur[st == 1]  # 当前帧中的跟踪点
    good_old = pts_prev[st == 1]  # 前一帧中的跟踪点
    # print('local points num:', len(good_old))
    if len(good_old) < 11:
        homography_matrix = np.array([[0.999, 0, 0], [0, 0.999, 0], [0, 0, 1]])
    else:
        homography_matrix, status = cv2.findHomography(good_new, good_old, cv2.RANSAC, 3.0)

    # 根据变换矩阵计算变换之后的图像
    compensated = cv2.warpPerspective(frame1, homography_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    # 计算掩膜
    vertex = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32).reshape(-1, 1, 2)
    homo_inv = np.linalg.inv(homography_matrix)
    vertex_trans = cv2.perspectiveTransform(vertex, homo_inv)
    vertex_transformed = np.array(vertex_trans, dtype=np.int32).reshape(1, 4, 2)
    im = np.zeros(frame1.shape[:2], dtype='uint8')
    cv2.polylines(im, vertex_transformed, 1, 255)
    cv2.fillPoly(im, vertex_transformed, 255)
    mask = 255 - im

    return compensated, mask


def frame_compensate(frame1, frame2):
    # grid-based KLT tracking
    lk_params = dict(winSize=(15, 15), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.003))

    # 创建随机生成的颜色
    color = np.random.randint(0, 255, (3000, 3))

    width = frame2.shape[1]
    height = frame2.shape[0]
    gridSizeW = 32 * 2
    gridSizeH = 24 * 2
    p1 = []
    grid_numW = int(width / gridSizeW - 1)
    grid_numH = int(height / gridSizeH - 1)
    for i in range(grid_numW):
        for j in range(grid_numH):
            point = (np.float32(i * gridSizeW + gridSizeW / 2.0), np.float32(j * gridSizeH + gridSizeH / 2.0))
            p1.append(point)

    p1 = np.array(p1)
    pts_num = grid_numW * grid_numH
    pts_prev = p1.reshape(pts_num, 1, 2)

    pts_cur, st, err = cv2.calcOpticalFlowPyrLK(frame1, frame2, pts_prev, None, **lk_params)

    # 选择good points
    good_new = pts_cur[st == 1]  # 当前帧中的跟踪点
    good_old = pts_prev[st == 1]  # 前一帧中的跟踪点

    homography_matrix, status = cv2.findHomography(good_new, good_old, cv2.RANSAC, 3.0)
    # homography_matrix, status = cv2.findHomography(points_new, points_old, cv2.RANSAC, 3.0)
    # matchesMask = status.ravel().tolist()

    # 根据变换矩阵计算变换之后的图像
    # compensated = cv2.warpPerspective(frame1, homography_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    #
    # # 计算掩膜
    # vertex = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32).reshape(-1, 1, 2)
    # homo_inv = np.linalg.inv(homography_matrix)
    # vertex_trans = cv2.perspectiveTransform(vertex, homo_inv)
    # vertex_transformed = np.array(vertex_trans, dtype=np.int32).reshape(1, 4, 2)
    # im = np.zeros(frame1.shape[:2], dtype='uint8')
    # cv2.polylines(im, vertex_transformed, 1, 255)
    # cv2.fillPoly(im, vertex_transformed, 255)
    # mask = 255 - im

    return homography_matrix


def enlargebox(x, y, w, h, a, width, height):
    # xa = int(w * a)
    # ya = int(h * a)
    xa = a
    ya = a
    x1 = x - xa
    y1 = y - ya
    w1 = w + xa * 2
    h1 = h + ya * 2

    if x1 < 0:
        x1 = 0

    if y1 < 0:
        y1 = 0

    if x1 + w1 >= width:
        w1 = width - x1 - 1

    if y1 + h1 >= height:
        h1 = height - y1 - 1

    return int(x1), int(y1), int(w1), int(h1)


def dist(x1, y1, x2, y2):
    distance = np.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))
    return distance


def rect_dist(x1, y1, w1, h1, x2, y2, w2, h2):
    # 转化为左上角和右下角坐标
    x1b = x1 + w1
    y1b = y1 + h1
    x2b = x2 + w2
    y2b = y2 + h2

    left = x2b < x1
    right = x1b < x2
    bottom = y2b < y1
    top = y1b < y2

    if top and left:
        return dist(x1, y1b, x2b, y2)
    elif left and bottom:
        return dist(x1, y1, x2b, y2b)
    elif bottom and right:
        return dist(x1b, y1, x2, y2b)
    elif right and top:
        return dist(x1b, y1b, x2, y2)
    elif left:
        return x1 - x2b
    elif right:
        return x2 - x1b
    elif bottom:
        return y1 - y2b
    elif top:
        return y2 - y1b
    else:  # rectangles intersect
        return 0


def two2one(x1, y1, w1, h1, x2, y2, w2, h2):
    """
    将两个矩形框，变成一个更大的矩形框
    input：两个矩形框，分别左上角和右下角坐标
    return：融合后矩形框左上角和右下角坐标
    """
    # 转化为左上角和右下角坐标
    x1b = x1 + w1
    y1b = y1 + h1
    x2b = x2 + w2
    y2b = y2 + h2

    x = min(x1, x2)
    y = min(y1, y2)
    xb = max(x1b, x2b)
    yb = max(y1b, y2b)

    return x, y, xb, yb


def box_select(boxes1):
    """
    多box，最终融合距离近的，留下新的，或未被融合的
    input：多box的列表，例如：[[12,23,45,56],[36,25,45,63],[30,25,60,35]]
    return：新的boxes，这里面返回的结果是这样的，被合并的box会置为[]，最终返回的，可能是这样[[],[],[50,23,65,50]]
    """

    # print("boxes1:", boxes1)
    if len(boxes1) > 0:
        for bi in range(len(boxes1)):
            for bj in range(len(boxes1)):
                if bi != bj:
                    if len(boxes1[bi]) == 4 and len(boxes1[bj]) == 4:
                        x1, y1, w1, h1 = int(boxes1[bi][0]), int(boxes1[bi][1]), int(boxes1[bi][2]), int(boxes1[bi][3])
                        x2, y2, w2, h2 = int(boxes1[bj][0]), int(boxes1[bj][1]), int(boxes1[bj][2]), int(boxes1[bj][3])

                        dis = rect_dist(x1, y1, w1, h1, x2, y2, w2, h2)
                        if dis < 10:
                            # print('merge boxes')
                            x, y, xb, yb = two2one(x1, y1, w1, h1, x2, y2, w2, h2)
                            boxes1[bj][0] = x
                            boxes1[bj][1] = y
                            boxes1[bj][2] = xb - x
                            boxes1[bj][3] = yb - y
                            boxes1[bi] = np.zeros(4)

    return boxes1


def resnet34_infer(src):
    # net = cv2.dnn.readNetFromONNX('/home/user-guo/Downloads/video/20220731/phantom_100m/far_sky/vgg16_1.onnx')
    net = cv2.dnn.readNetFromONNX('./mydataset/resnet34_1.onnx')
    size = 224
    image = cv2.resize(src, [size, size])
    blob = cv2.dnn.blobFromImage(image, 1.0, (size, size), (0, 0, 0), False)
    net.setInput(blob)
    probs = net.forward()
    index = np.argmax(probs)
    return index


class MyNet(nn.Module):

    def __init__(self, num_classes=2) -> None:
        super(MyNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(1024, 64),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.model(x)
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def Mynet_infer(src):
    # data_transform = transforms.Compose([transforms.ToTensor()])
    data_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([32, 32])])
    # size = 32
    # img = cv2.resize(src, (size, size))
    img = data_transform(src)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Net()
    # load model weights
    # model_weight_path = "/home/user-guo/Documents/MovingDrone/weight/Net_best_1.pth"
    model_weight_path = "./weight/GLAD2_best.pth"
    # model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cpu')))
    # model.to(device)
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img))
        predict = torch.softmax(output, dim=0)
        score = predict.numpy()[1]
        predict_cla = torch.argmax(predict).numpy()
        # print(predict_cla)

    return predict_cla, score


def Net_infer(src, net):
    blob = cv2.dnn.blobFromImage(src, 1.0, (32, 32), (0, 0, 0), False)
    net.setInput(blob)
    probs = net.forward()
    index = np.argmax(probs)

    return index


def Net_onnx(src, session):
    # session = ort.InferenceSession('/home/user-guo/Documents/MovingDrone/weight/Net_best_1.onnx')
    data_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize([32, 32])])
    image = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    input_tensor = data_transform(image)
    input_tensor = input_tensor.unsqueeze(0)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor.numpy()})

    predicted_class_idx = np.argmax(outputs[0])

    return predicted_class_idx


def resnet18_infer(src):
    data_transform = transforms.Compose([transforms.ToTensor()])
    size = 32
    img = cv2.resize(src, (size, size))
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model = resnet18()
    # load model weights
    model_weight_path = "/home/user-guo/Documents/MovingDrone/weight/resnet18_best_1.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cuda')))
    # model.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img))
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
        # print(predict_cla)

    return predict_cla


def readGT(file, image_id):
    in_file = open(file + 'frame' + '%s.xml' % image_id, encoding='UTF-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    box = []
    for obj in root.iter('object'):
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        # 标注越界修正
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h

        x0 = b1
        y0 = b3
        w0 = b2 - b1
        h0 = b4 - b3

        box0 = [x0, y0, w0, h0]
        box.append(box0)
    box = np.array(box)

    return box


def enlarge_region2(x, y, a, width, height):
    x1 = x - a
    y1 = y - a
    w1 = a * 2
    h1 = a * 2

    if x1 < 0:
        x1 = 0

    if y1 < 0:
        y1 = 0

    if x1 + w1 >= width:
        x1 = width - w1

    if y1 + h1 >= height:
        y1 = height - h1

    return int(x1), int(y1), int(w1), int(h1)


def cal_iou(box1, box2):
    """

    :param box1: xywh 左上右下
    :param box2: xywh
    :transfer to xyxy
    """
    x1min, y1min, x1max, y1max = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
    x2min, y2min, x2max, y2max = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]
    # 计算两个框的面积
    s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
    s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)

    # 计算相交部分的坐标
    xmin = max(x1min, x2min)
    ymin = max(y1min, y2min)
    xmax = min(x1max, x2max)
    ymax = min(y1max, y2max)

    inter_h = max(ymax - ymin + 1, 0)
    inter_w = max(xmax - xmin + 1, 0)

    intersection = inter_h * inter_w
    union = s1 + s2 - intersection

    # 计算iou
    iou = intersection / union
    return iou


def readGTbox(xml_file):
    global x3, y3, w3, h3, GT_box
    tree = ET.parse(xml_file)
    root = tree.getroot()

    if root.find('object') == None:
        GT_box = []
        return GT_box
    else:
        for obj in root.iter('object'):
            xmlbox = obj.find('bndbox')
            b = [int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)),
                 int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text))]

            x3 = b[0]
            y3 = b[1]
            w3 = b[2] - b[0]
            h3 = b[3] - b[1]

            GT_box = np.array([x3, y3, w3, h3])

        return GT_box