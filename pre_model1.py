# 导入所需的库
import json
import shutil
import time
import warnings

warnings.filterwarnings("ignore")  # 忽略警告信息
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from music21 import *
from midi2audio import FluidSynth
import os
from skimage.transform import hough_line, rotate, hough_line_peaks
from skimage.feature import corner_harris
from skimage.measure import label, regionprops
from skimage.color import label2rgb
from skimage.exposure import histogram
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu, gaussian, median
from skimage.morphology import binary_opening, binary_closing, binary_dilation, binary_erosion, opening, \
    square, disk
from skimage.feature import canny
from matplotlib.pyplot import bar
from scipy.ndimage import binary_fill_holes
from skimage.morphology import thin
import cv2
import math
import imutils
from flask import Flask, render_template
from flask import request, jsonify
from flask_cors import CORS, cross_origin
from gevent.greenlet import Greenlet
from gevent.pywsgi import WSGIServer
import skimage.io as io
from mido import Message, MidiFile, MidiTrack
import librosa

plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文乱码

headers = {'Content-Type': 'application/json'}
# 定义了一个二元组，表示图像目标大小为 100x100 像素
target_img_size = (100, 100)
# 定义了一个整数变量，表示样本数为 50
sample_count = 50
# 字典用于将标签映射到特定的字符串表示形式，其中 key 为标签，value 是一个嵌套字典，该字典的每个 key 是类别编号，对应的 value 是类别名称字符串
label_map = {
    0: {
        0: 'N0'
    },
    1: {
        0: 'b2',
        1: 'a2'
    },
    2: {
        0: 'g2',
        1: 'f2'
    },
    3: {
        0: 'e2',
        1: 'd2'
    },
    4: {
        0: 'c2',
        1: 'b1'
    },
    5: {
        0: 'a1',
        1: 'g1'
    },
    6: {
        0: 'f1',
        1: 'e1'
    },
    7: {
        0: 'd1',
        1: 'c1'
    }
}
# 定义了一个浮点数变量，它表示每个标签在样本数据集中的占比，该值为 0.3，即每个标签占样本数据集的 30%
row_percentage = 0.3


class Segmenter(object):

    def __init__(self, bin_img):   # Python 类的构造函数，当创建 Segmenter 对象时自动调用。
        self.bin_img = bin_img     # 传入参数是二值化图像 bin_img，因此在 __init__ 函数中存储该图像（self.bin_img）
        self.rle, self.vals = hv_rle(self.bin_img)   # 使用 hv_rle 函数将其行程长度编码表示（self.rle 和 self.vals）
        self.most_common = get_most_common(self.rle)   # 存储self.most_common 变量
        self.thickness, self.spacing = calculate_thickness_spacing(      # 然后计算一个常见的杆的宽度和间距（self.thickness 和 self.spacing）
            self.rle, self.most_common)
        self.thick_space = self.thickness + self.spacing
        self.no_staff_img = remove_staff_lines(                       # 使用 remove_staff_lines 函数从二值化图像中移除五线谱
            self.rle, self.vals, self.thickness, self.bin_img.shape)  # 得到一个没有五线谱的图像（self.no_staff_img）
        self.segment()                                # 调用 segment() 函数进行分割处理。

    #这个函数根据传入的区域大小（region），使用尺寸为 thickness x thickness 的矩形结构元素进行形态学开操作，以去除区域内的小噪声。
    def open_region(self, region):
        thickness = np.copy(self.thickness)
        # if thickness % 2 == 0:
        #     thickness += 1
        return opening(region, np.ones((thickness, thickness)))
    # 这是 Segmenter 类的分割函数，根据输入二值化图像中检测到的五线谱将其分割成板块。。，，。
    def segment(self):
        self.line_indices = get_line_indices(histogram(self.bin_img, 0.8))    #首先使用 histogram 函数确定输入图像中的线的位置，然后使用 get_line_indices 函数返回这些线的 y 坐标
        if len(self.line_indices) < 10:            #如果检测到的线的数量少于10根，说明输入图像中不含五线谱
            self.regions_without_staff = [
                np.copy(self.open_region(self.no_staff_img))]    #通过 open_region 函数对输入图像处理，将其存储在 regions_without_staff 矩阵中
            self.regions_with_staff = [np.copy(self.bin_img)]    #使用 bin_img 存储原始输入图像，然后退出函数
            return
    #对于检测到五线谱的情况，找到前两个线之间的距离用于估计行的间距，根据上下五线之间的距离确定五线谱的宽度。计算距离前两根线最近的行的位置，除以 2 得到行的中心。最后返回每个行之间的距离 (spacing_between_staff_blocks) 和每个盒子的中心 (box_centers)。
        generated_lines_img = np.copy(self.no_staff_img)
        lines = []
        for index in self.line_indices:
            line = ((0, index), (self.bin_img.shape[1] - 1, index))
            lines.append(line)
    # 将原始图像中的五线谱区域分割成若干个小块，并将每个小块进行预处理以便后续的符号识别
        end_of_staff = []
        for index, line in enumerate(lines):
            if index > 0 and (line[0][1] - end_of_staff[-1][1] < 4 * self.spacing):
                pass
            else:
                p1, p2 = line
                x0, y0 = p1
                x1, y1 = p2
                end_of_staff.append((x0, y0, x1, y1))

        box_centers = []
        spacing_between_staff_blocks = []
        for i in range(len(end_of_staff) - 1):
            spacing_between_staff_blocks.append(
                end_of_staff[i + 1][1] - end_of_staff[i][1])
            if i % 2 == 0:
                offset = (end_of_staff[i + 1][1] - end_of_staff[i][1]) // 2
                center = end_of_staff[i][1] + offset
                box_centers.append((center, offset))

        max_staff_dist = np.max(spacing_between_staff_blocks)
        max_margin = max_staff_dist // 2
        margin = max_staff_dist // 10

        end_points = []
        regions_without_staff = []
        regions_with_staff = []
        for index, (center, offset) in enumerate(box_centers):
            y0 = int(center) - max_margin - offset + margin
            y1 = int(center) + max_margin + offset - margin
            end_points.append((y0, y1))

            region = self.bin_img[y0:y1, 0:self.bin_img.shape[1]]
            regions_with_staff.append(region)
            staff_block = self.no_staff_img[y0:y1,
                          0:self.no_staff_img.shape[1]]

            regions_without_staff.append(self.open_region(staff_block))

        self.regions_without_staff = regions_without_staff
        self.regions_with_staff = regions_with_staff


def extract_raw_pixels(img):        #将输入的图像调整为预定义的尺寸（为target_img_size），然后将像素展平为一维数组并返回
    resized = cv2.resize(img, target_img_size)
    return resized.flatten()


def extract_hsv_histogram(img):     #将输入的图像调整为预定义的尺寸，然后将其转换为HSV颜色空间并计算该图像的3D直方图。这个直方图可以提供一个用于特征描述的小向量。
    resized = cv2.resize(img, target_img_size)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8],
                        [0, 180, 0, 256, 0, 256])
    if imutils.is_cv2():
        hist = cv2.normalize(hist)
    else:
        cv2.normalize(hist, hist)
    return hist.flatten()


def extract_hog_features(img):    # 计算一个图像的方向梯度直方图（HOG）特征，这种特征可以使用分类器进行训练和分类。该函数先将输入图像调整为预定义的尺寸，然后使用cv2.HOGDescriptor对象来计算图像的HOG特征
    img = cv2.resize(img, target_img_size)
    win_size = (100, 100)
    cell_size = (4, 4)
    block_size_in_cells = (2, 2)

    block_size = (block_size_in_cells[1] * cell_size[1],
                  block_size_in_cells[0] * cell_size[0])
    block_stride = (cell_size[1], cell_size[0])
    nbins = 9  # Number of orientation bins
    hog = cv2.HOGDescriptor(win_size, block_size,
                            block_stride, cell_size, nbins)
    h = hog.compute(img)
    h = h.flatten()
    return h.flatten()


def extract_features(img, feature_set='raw'):   #通用的特征提取函数，根据所提供的参数来决定使用哪种特征集或方法。如果feature_set参数为hog，返回HOG特征；如果参数为raw，返回展平后的像素值。否则，返回HSV直方图特征
    if feature_set == 'hog':
        return extract_hog_features(img)
    elif feature_set == 'raw':
        return extract_raw_pixels(img)
    else:
        return extract_hsv_histogram(img)


def rle_encode(arr):   # 函数实现了行程长度编码（Run-length encoding，RLE）算法，经常用于图像压缩技术和视频编码中。传入一个一维数组，函数输出该数组的RLE编码结果，以便于传输或存储
    if len(arr) == 0:
        return [], [], []

    x = np.copy(arr)
    first_dismatch = np.array(x[1:] != x[:-1])
    distmatch_positions = np.append(np.where(first_dismatch), len(x) - 1)
    rle = np.diff(np.append(-1, distmatch_positions))
    values = [x[i] for i in np.cumsum(np.append(0, rle))[:-1]]
    return rle, values

# 以下两个函数用于对二值图像进行水平和垂直方向的行程长度编码（Run-length encoding）和解码操作
def hv_rle(img, axis=1):    # 该函数用于对输入的二值图像进行行程长度编码。axis是一个可选的参数，将其设置为1将对列进行编码，设置为0将对行进行编码。函数通过执行rle_encode子函数实现编码，并将编码结果存储在两个列表中，分别代表编码结果和编码值。最后，函数返回这两个列表。
    '''
    img: binary image
    axis: 0 for rows, 1 for cols
    '''
    rle, values = [], []

    if axis == 1:
        for i in range(img.shape[1]):
            col_rle, col_values = rle_encode(img[:, i])
            rle.append(col_rle)
            values.append(col_values)
    else:
        for i in range(img.shape[0]):
            row_rle, row_values = rle_encode(img[i])
            rle.append(row_rle)
            values.append(row_values)

    return rle, values


def rle_decode(starts, lengths, values):   # 该函数对输入的编码数据进行解码操作，返回原始数组。函数将输入参数转换为NumPy数组，并利用编码的起始索引、长度和值来重构原始数组。函数首先计算数组的长度，并使用np.full创建一个具有NaN填充值的数组，最后使用zip函数来遍历每一个编码字段，并在目标数组中填充值。函数返回完成解码操作的目标数组
    starts, lengths, values = map(np.asarray, (starts, lengths, values))
    ends = starts + lengths
    n = ends[-1]

    x = np.full(n, np.nan)
    for lo, hi, val in zip(starts, ends, values):
        x[lo:hi] = val
    return x

# 以下是一些与解码二值图像编码格式相关的辅助函数
def hv_decode(rle, values, output_shape, axis=1):   #函数将行程长度编码数据转换为原始图像形式。该函数提取各个编码数据中的起始点，然后在二维数组decoded中迭代列或行，解码每个列或行的编码数据。生成的解码结果将是形状与output_shape相同的NumPy数组
    starts = [[int(np.sum(arr[:i])) for i in range(len(arr))] for arr in rle]

    decoded = np.zeros(output_shape, dtype=np.int32)
    if axis == 1:
        for i in range(decoded.shape[1]):
            decoded[:, i] = rle_decode(starts[i], rle[i], values[i])
    else:
        for i in range(decoded.shape[0]):
            decoded[i] = rle_decode(starts[i], rle[i], values[i])

    return decoded


def calculate_pair_sum(arr):    #函数计算传入数组的相邻元素的和，组成一个列表并返回。该函数首先检查数组的长度，如果只有一个元素，则将其返回。否则，它将使用一行代码来计算并返回该列表，首先对相邻的两个元素求和，然后将它们加入到数组中，使用步长为2以跳过一个元素
    if len(arr) == 1:
        return list(arr)
    else:
        res = [arr[i] + arr[i + 1] for i in range(0, len(arr) - 1, 2)]
        if len(arr) % 2 == 1:
            res.append(arr[-2] + arr[-1])
        return res


def get_most_common(rle):  # 函数计算编码数据中的常见黑白像素对和值。该函数首先对每列的像素对进行计算并生成列表pair_sum，该列表包含传输数组的每列的相邻元素的和。接下来，该函数将所有列的坐标和值压平到一个列表中，并使用np.bincount()函数计算最常见的像素对。函数返回最常见的像素对值。
    pair_sum = [calculate_pair_sum(col) for col in rle]

    flattened = []
    for col in pair_sum:
        flattened += col

    most_common = np.argmax(np.bincount(flattened))
    return most_common


def most_common_bw_pattern(arr, most_common):  # 函数查找传递的数组中是否存在与最常见的黑白像素对值相匹配的像素对。该函数首先检查数组的长度，如果数组长仅为1，则返回一个空列表。否则，函数使用一行代码对每一对相邻的像素进行求和，并将结果与“最常见的像素对”进行比较。匹配的像素对以元组形式添加到列表res中并返回
    if len(arr) == 1:
        # print("Empty")
        return []
    else:
        res = [(arr[i], arr[i + 1]) for i in range(0, len(arr) - 1, 2)
               if arr[i] + arr[i + 1] == most_common]

        if len(arr) % 2 == 1 and arr[-2] + arr[-1] == most_common:
            res.append((arr[-2], arr[-1]))
        # print(res)
        return res

# 定义了一个名为Box的类，它表示一个矩形框
class Box(object):
    def __init__(self, x, y, w, h):
        self.x = x   # x表示矩形框左上角的x坐标
        self.y = y    # y表示矩形框左上角的y坐标
        self.w = w    # w表示矩形框的宽度
        self.h = h    # h表示矩形框的高度
        self.center = x + w / 2, self.y + self.h / 2      # center表示矩形框的中心点坐标（由x+w/2和y+h/2计算得到）
        self.area = w * h      # area表示矩形框的面积（由w*h计算得到）

    # overlap函数接受另一个Box对象作为参数，计算两个矩形框之间的重叠面积，并将其除以当前矩形框的面积，最终返回一个小于或等于1的值表示覆盖率（intersection over union）
    def overlap(self, other):
        x = max(0, min(self.x + self.w, other.x + other.w) - max(other.x, self.x))
        y = max(0, min(self.y + self.h, other.y + other.h) - max(other.y, self.y))
        area = x * y
        return area / self.area
    # distance函数接受另一个Box对象作为参数，计算两个矩形框之间的中心点距离
    def distance(self, other):
        return math.sqrt((self.center[0] - other.center[0]) ** 2 + (self.center[1] - other.center[1]) ** 2)
    # merge函数接受另一个Box对象作为参数，返回一个新的Box对象，该对象表示两个矩形框的合并区域
    def merge(self, other):
        x = min(self.x, other.x)
        y = max(self.y, other.y)
        w = max(self.x + self.w, other.x + other.w) - x
        h = max(self.y + self.h, other.y + other.h) - y
        return Box(x, y, w, h)
    # draw函数接受一个图像、颜色和线条粗细作为参数，将当前Box对象绘制为一个矩形框并将其添加到图像中
    def draw(self, img, color, thickness):
        pos = ((int)(self.x), (int)(self.y))
        size = ((int)(self.x + self.w), (int)(self.y + self.h))
        cv2.rectangle(img, pos, size, color, thickness)

# show_images接受一个图像列表和对应的标题，用matplotlib显示每个图像。当没有提供标题时，它使用默认的标题。最后，它设置了图像显示区域的大小并显示图像
def show_images(images, titles=None):
    n_ims = len(images)
    if titles is None:
        titles = ['(%d)' % i for i in range(1, n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image, title in zip(images, titles):
        a = fig.add_subplot(1, n_ims, n)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        plt.axis('off')
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()


def showHist(img):  #showHist函数：接收一个图像作为参数，将该图像的灰度直方图绘制出来
    plt.figure()
    imgHist = histogram(img, nbins=256)

    bar(imgHist[1].astype(np.uint8), imgHist[0], width=0.8, align='center')


def gray_img(img): # gray_img函数：将RGB图像转换为灰度图像并返回。如果图像是RGB格式，则在转换之后还将像素值乘以255
    '''
    img: rgb image
    return: gray image, pixel values 0:255
    '''
    img = img[:, :, :3]
    gray = rgb2gray(img)
    if len(img.shape) == 3:
        gray = gray * 255
    return gray


def otsu(img):  #otsu函数：基于高斯模糊后的灰度图像，使用Otsu二值化算法将图像转换为二进制格式后返回
    '''
    Otsu with gaussian
    img: gray image
    return: binary image, pixel values 0:1
    '''
    blur = gaussian(img)
    otsu_bin = 255 * (blur > threshold_otsu(blur))
    return (otsu_bin / 255).astype(np.int32)


def get_gray(img): # get_gray函数：将RGB格式的图像转换为灰度图像并返回
    gray = rgb2gray(np.copy(img))
    return gray


def get_thresholded(img, thresh):  # get_thresholded函数：接收灰度图像和阈值作为参数，将像素大于阈值的区域设置为1，其余区域设置为0。返回一个二进制格式的图像
    return 1 * (img > thresh)


def histogram(img, thresh):   # histogram函数：接收灰度图像和阈值作为参数，返回一维数组，该数组表示每一行的黑色像素个数。如果像素个数小于阈值的最大值，则将一个像素的数量设置为0。返回结果为一维数组
    hist = (np.ones(img.shape) - img).sum(dtype=np.int32, axis=1)
    _max = np.amax(hist)
    hist[hist[:] < _max * thresh] = 0
    return hist


def get_line_indices(hist):   # get_line_indices函数：接收histogram函数的输出结果作为参数，返回一个列表，其中包含黑色像素数量出现变化的位置（行数）
    indices = []
    prev = 0
    for index, val in enumerate(hist):
        if val > 0 and prev <= 0:
            indices.append(index)
        prev = val
    return indices


def get_region_lines_indices(self, region): #get_region_lines_indices方法：接收一个包含音符区域的图像区域，获取区域内的乐谱行，并将其添加到类的“rows”列表中。乐谱行是在区域的灰度值直方图中基于阈值的黑/白像素分布而获得的
    indices = get_line_indices(histogram(region, 0.8))
    lines = []
    for line_index in indices:
        line = []
        for k in range(self.thickness):
            line.append(line_index + k)
        lines.append(line)
    self.rows.append([np.average(x) for x in lines])


def calculate_thickness_spacing(rle, most_common): #calculate_thickness_spacing方法：接收一个RLE编码和最常见的黑/白像素模式，计算乐谱中线条的宽度和间隔高度。它首先获取所有列的黑/白像素模式，然后返回最常见的一对模式出现的数量，其中更小的数字是线条厚度，更大的数字是行间距
    bw_patterns = [most_common_bw_pattern(col, most_common) for col in rle]
    bw_patterns = [x for x in bw_patterns if x]  # Filter empty patterns

    flattened = []
    for col in bw_patterns:
        flattened += col

    pair, count = Counter(flattened).most_common()[0]

    line_thickness = min(pair)
    line_spacing = max(pair)

    return line_thickness, line_spacing


def whitene(rle, vals, max_height): # whitene方法：根据最大高度将黑色字符像素（线条）替换为白色（背景），并返回更改后的RLE编码和黑/白值，这些值被用于计算后续乐谱行的空间
    rlv = []
    for length, value in zip(rle, vals):
        if value == 0 and length < 1.1 * max_height:
            value = 1
        rlv.append((length, value))

    n_rle, n_vals = [], []
    count = 0
    for length, value in rlv:
        if value == 1:
            count = count + length
        else:
            if count > 0:
                n_rle.append(count)
                n_vals.append(1)

            count = 0
            n_rle.append(length)
            n_vals.append(0)
    if count > 0:
        n_rle.append(count)
        n_vals.append(1)

    return n_rle, n_vals


def remove_staff_lines(rle, vals, thickness, shape): # remove_staff_lines方法：从二值图像的RLE编码和黑/白像素值中删除乐谱行，并将结果解码为二进制图像格式。该方法使用whitene将乐谱行转换为白色空白区域，以便它们可以被filter_staff_lines2方法过滤掉
    n_rle, n_vals = [], []
    for i in range(len(rle)):
        rl, val = whitene(rle[i], vals[i], thickness)
        n_rle.append(rl)
        n_vals.append(val)

    return hv_decode(n_rle, n_vals, shape)


def remove_staff_lines_2(thickness, img_with_staff): # remove_staff_lines_2方法：根据乐谱线条的间距（使用calculate_thickness_spacing方法确定），从二进制图像中过滤掉乐谱行。涉及将图像均匀分成许多行，然后计算每行中值为1的像素数量。如果此计数低于行间距的阈值，则将该行过滤掉
    img = img_with_staff.copy()
    projected = []
    rows, cols = img.shape
    for i in range(rows):
        proj_sum = 0
        for j in range(cols):
            proj_sum += img[i][j] == 1
        projected.append([1] * proj_sum + [0] * (cols - proj_sum))
        if (proj_sum <= row_percentage * cols):
            img[i, :] = 1
    closed = binary_opening(img, np.ones((3 * thickness, 1)))
    return closed

# 此部分为音乐OCR(光学乐谱识别)项目取一个图像作为输入并返回音符等元素在图像中的位置。
# 函数目的是进行图像处理，以从输入图像中提取有用的信息以识别音符和其他要素
def get_rows(start, most_common, thickness, spacing):  #此函数返回一个二维数组，其中包含从起始位置开始thickenss个元素的列表，这些元素之间间隔为spacing。变量start 的初始值为 start -most_common。这个函数的目的是根据乐谱的起点和最常用空间计算得出每个小节的位置。
    rows = []
    num = 6
    if start - most_common >= 0:
        start -= most_common
        num = 7
    for k in range(num):
        row = []
        for i in range(thickness):
            row.append(start)
            start += 1
        start += (spacing)
        rows.append(row)
    if len(rows) == 6:
        rows = [0] + rows
    return rows


def horizontal_projection(img):  #此函数用于将垂直像素投影成一个数组。投影计算是根据每个垂直像素的黑白值计算的。该函数还包括在二值输入图像中检测到空白行的条件。
    projected = []
    rows, cols = img.shape
    for i in range(rows):
        proj_sum = 0
        for j in range(cols):
            proj_sum += img[i][j] == 1
        projected.append([1] * proj_sum + [0] * (cols - proj_sum))
        if (proj_sum <= 0.1 * cols):
            return i
    return 0


def get_staff_row_position(img):  # 此函数用于获取第一个非白行的行位置。该函数用于确定图像中行的位置，有助于去除谱表上的行，以便从输入图像中提取有用信息。
    found = 0
    row_position = -1
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (img[i][j] == 0):
                row_position = i
                found = 1
                break
        if found == 1:
            break
    return row_position


def coordinator(bin_img, horizontal):  # 此函数使用各个处理的子函数来定位谱表的行和乐符，将它们作为numpy数组返回。 coordinator() 函数还计算出小节起点，并输出二值化列表，其中文件包含非谱表的元素
    rle, vals = hv_rle(bin_img)
    most_common = get_most_common(rle)
    thickness, spacing = calculate_thickness_spacing(rle, most_common)
    start = 0
    if horizontal:
        no_staff_img = remove_staff_lines_2(thickness, bin_img)
        staff_lines = otsu(bin_img - no_staff_img)
        start = horizontal_projection(bin_img)
    else:
        no_staff_img = remove_staff_lines(rle, vals, thickness, bin_img.shape)
        no_staff_img = binary_closing(
            no_staff_img, np.ones((thickness + 2, thickness + 2)))
        no_staff_img = median(no_staff_img)
        no_staff_img = binary_opening(
            no_staff_img, np.ones((thickness + 2, thickness + 2)))
        staff_lines = otsu(bin_img - no_staff_img)
        staff_lines = binary_erosion(
            staff_lines, np.ones((thickness + 2, thickness + 2)))
        staff_lines = median(staff_lines, selem=square(21))
        start = get_staff_row_position(staff_lines)
    staff_row_positions = get_rows(
        start, most_common, thickness, spacing)
    staff_row_positions = [np.average(x) for x in staff_row_positions]
    return spacing, staff_row_positions, no_staff_img


def deskew(image):  # deskew(image)：这个函数用于反斜校正，即将旋转的图像矫正为水平方向。在函数中，首先使用Canny算法找到图像中的边缘，然后使用 Harris 角点检测算法检测边缘中的角点。接着，使用霍夫直线变换求出图像中的直线，并通过平均角度计算旋转角度。如果旋转角度小于45度且不为0，则将其加上90度并返回
    edges = canny(image, low_threshold=50, high_threshold=150, sigma=2)
    harris = corner_harris(edges)
    tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
    h, theta, d = hough_line(harris, theta=tested_angles)
    out, angles, d = hough_line_peaks(h, theta, d)
    rotation_number = np.average(np.degrees(angles))
    if rotation_number < 45 and rotation_number != 0:
        rotation_number += 90
    return rotation_number


def rotation(img, angle):  # 这个函数对输入的图像进行旋转。使用Rotate函数进行图像旋转，并通过mode='edge’参数表示边缘像素值采用扩充模式
    image = rotate(img, angle, resize=True, mode='edge')
    return image


def get_closer(img): # 这个函数用于对图像进行裁剪以消除多余的空白边缘。首先将图像分成16块，计算每块中黑色像素点的数量。如果黑色像素占该块像素总数的1%，则说明该块有音乐符号，记录该块的行和列。最后根据行和列的记录，裁剪出包含所有音乐符号的最小矩形
    rows = []
    cols = []
    for x in range(16):
        no = 0
        for col in range(x * img.shape[0] // 16, (x + 1) * img.shape[0] // 16):
            for row in range(img.shape[1]):
                if img[col][row] == 0:
                    no += 1
        if no >= 0.01 * img.shape[1] * img.shape[0] // 16:
            rows.append(x * img.shape[0] // 16)
    for x in range(16):
        no = 0
        for row in range(x * img.shape[1] // 16, (x + 1) * img.shape[1] // 16):
            for col in range(img.shape[0]):
                if img[col][row] == 0:
                    no += 1
        if no >= 0.01 * img.shape[0] * img.shape[1] // 16:
            cols.append(x * img.shape[1] // 16)
    new_img = img[rows[0]:min(img.shape[0], rows[-1] + img.shape[0] // 16),
              cols[0]:min(img.shape[1], cols[-1] + img.shape[1] // 16)]
    return new_img


def IsHorizontal(img):   # IsHorizontal(img)：这个函数判断输入的图像是否为水平的，即是否为横谱。通过计算每一行黑色像素点的数量，判断是否有一行黑色像素的比例超过总像素的90%，如果有，说明该图像为水平的
    projected = []
    rows, cols = img.shape
    for i in range(rows):
        proj_sum = 0
        for j in range(cols):
            if img[i][j] == 0:
                proj_sum += 1
        projected.append([1] * proj_sum + [0] * (cols - proj_sum))
        if (proj_sum >= 0.9 * cols):
            return True
    return False


def get_connected_components(img_without_staff, img_with_staff):  #get_connected_components(img_without_staff, img_with_staff)：这个函数用于获取输入图像中的连通组件。首先使用大津算法（Otsu）获取二值化的图像阈值，然后通过闭运算处理得到二进制图像。接着使用skimage库中的label函数标记出连通区域并返回。
    components = []
    boundary = []
    # thresh = threshold_otsu(img_without_staff)
    # bw = closing(img_without_staff <= thresh, square(3))
    bw = 1 - img_without_staff
    label_img = label(bw)
    img_label_overlay = label2rgb(
        label_img, image=img_without_staff, bg_label=0)
    for region in regionprops(label_img):
        if region.area >= 100:
            boundary.append(region.bbox)

    boundary = sorted(boundary, key=lambda b: b[1])

    comp_with_staff = []
    for bbox in boundary:
        minr, minc, maxr, maxc = bbox
        components.append(img_without_staff[minr:maxr, minc:maxc])
        comp_with_staff.append(img_with_staff[minr:maxr, minc:maxc])
    return components, comp_with_staff, boundary


def estim(c, idx, imgs_spacing, imgs_rows): # estim(c, idx, imgs_spacing, imgs_rows)：这个函数用于估计一个音符所在行和列。输入参数为该音符中心点的坐标 c、该音符所在图像的下标 idx、该图像的行间距 imgs_spacing 和所有行的位置信息 imgs_rows。根据音符中心点坐标 c 和行间距，计算出该音符所在的行数和偏移量（左侧还是右侧）
    spacing = imgs_spacing[idx]
    rows = imgs_rows[idx]
    margin = 1 + (spacing / 4)
    for index, line in enumerate(rows):
        if c >= line - margin and c <= line + margin:
            return index + 1, 0
        elif c >= line + margin and c <= line + 3 * margin:
            return index + 1, 1
    return 7, 1


def get_note_name(prev, octave, duration):  # get_note_name(prev, octave, duration)：这个函数用于获取音符的名称。输入参数为前一个音符的名称 prev、音符所在八度 octave 和音符时值（即音符符号的形状） duration。根据音符时值，返回该音符的名称
    if duration in ['4', 'a_4']:
        return f'{octave[0]}{prev}{octave[1]}/4'
    elif duration in ['8', '8_b_n', '8_b_r', 'a_8']:
        return f'{octave[0]}{prev}{octave[1]}/8'
    elif duration in ['16', '16_b_n', '16_b_r', 'a_16']:
        return f'{octave[0]}{prev}{octave[1]}/16'
    elif duration in ['32', '32_b_n', '32_b_r', 'a_32']:
        return f'{octave[0]}{prev}{octave[1]}/32'
    elif duration in ['2', 'a_2']:
        return f'{octave[0]}{prev}{octave[1]}/2'
    elif duration in ['1', 'a_1']:
        return f'{octave[0]}{prev}{octave[1]}/1'
    else:
        return "c1/4"


def filter_beams(prims, prim_with_staff, bounds): # filter_beams(prims, prim_with_staff, bounds)：这个函数用于过滤掉横杠而不是音符的图像。输入参数为所有图像的列表 prims、包含五线谱线的图像列表 prim_with_staff 和所有图像的边界框坐标信息 bounds。遍历每个图像，如果图像的宽度大于等于2倍的高度，则认为该图像是横杠而不是音符，将其过滤掉。最终返回过滤后的三个列表
    n_bounds = []
    n_prims = []
    n_prim_with_staff = []
    for i, prim in enumerate(prims):
        if prim.shape[1] >= 2 * prim.shape[0]:
            continue
        else:
            n_bounds.append(bounds[i])
            n_prims.append(prims[i])
            n_prim_with_staff.append(prim_with_staff[i])
    return n_prims, n_prim_with_staff, n_bounds

# 接受一个装有音符列表的参数 chord_list，然后将这个列表转换成一个特定的字符串格式返回
def get_chord_notation(chord_list):  #初始化一个字符串变量 chord_res，表示转换后的结果
    chord_res = "{"
    for chord_note in chord_list:   # 遍历输入的 chord_list，将其中的每个音符转换成字符串并添加到 chord_res 中。
        chord_res += (str(chord_note) + ",")
    chord_res = chord_res[:-1]
    chord_res += "}"

    return chord_res   #返回最终的 chord_res 字符串格式

# 一个包含字符串元素的列表，每个字符串都代表了一个音符或符号的标记。这些标记通常是由计算机识别或生成的，以帮助表示音乐的音符和节奏
# 标记说明：
    # bar：小节线
    # t44：指4/4拍子的节拍符号
    # 8：八分音符
    # a：A音的音符（A是一个例子，实际上可以是任何音符）
    # natural：自然记号，用于取消先前的升降记号
    # flat：降记号
    #：升记号
    # dot：附点，用于使音符持续时间增加一半
    # p：休止符，表示静默
label_str = ['bar_121', '32_b_r_049', '8_b_n_017', 't44_200', '8_b_r_021', 'bar_b_128', 't44_b_196', 'flat_b_165',
             '8_013', 'a_16_091', 't24_b_193', 'a_4_071', 'flat_b_163', '32_036', 't4_186', 'bar_b_125', 'a_1_050',
             'dot_b_152', 'a_2_062', 'a_2_058', 'bar_119', 'a_2_060', '32_b_r_043', 'a_8_079', 'p_180', 'natural_169',
             'bar_099', 'bar_111', 'flat_b_164', '#', 't2_185', 'bar_106', 'natural_b_170', 'bar_115', 't44_197',
             'a_4_068', 'a_2_056', 'natural_167', 'dot_147', '#_b_007', 'natural_b_169', '2_006', '8_015', 't24_191',
             'a_8_077', 'a_16_092', 'a_8_086', 'natural_165', 'a_2_059', '1_001', 'a_4_076', 'a_1_049', 'a_4_066',
             't44_198', '16_022', 'a_2_066', 'natural_b_174', 't44_201', '#_b_002', 't24_192', 't24_190', 'flat_157',
             'clef_135', 'natural_168', 't24_b_195', '8_014', 'natural_b_177', 'bar_098', 'a_8_083', 'a_32_096',
             'a_8_082', '8_b_r_019', '32_b_n_042', 'clef_b_137', 'a_1_051', 'bar_123', 'bar_107', 'a_16_086', 't2_184',
             '#_b_009', 'bar_101', 't44_199', 'a_1_052', 'bar_109', '2_007', '32_037', '2_008', 't24_b_194', 't2_181',
             'bar_b_129', '16_024', '16_b_r_030', '32_b_r_046', 'chord_132', '16_b_r_036', 'flat_156', '32_038',
             '32_b_r_047', 'a_8_080', 'a_1_055', 'clef_137', 'a_8_081', '16_b_n_025', 'a_16_087', 'bar_125',
             '16_b_r_031', '32_b_n_041', 'a_2_063', 'a_4_072', 'a_4_074', 'bar_103', '16_b_r_033', 'p_179', 't44_b_195',
             'dot_150', '#_b_001', '16_b_n_027', '32_b_n_039', 'flat_b_161', 'clef_136', 't4_189', 't2_183', '#_2',
             'a_8_084', 'natural_166', 'bar_b_127', 't24_193', 'clef_b_138', 'flat_b_160', 'flat_b_162', 'dot_149',
             'bar_097', 'clef_b_140', 'a_8_085', 'a_4_069', 'dot_b_151', '16_b_r_035', 'a_2_064', 'clef_b_141', '4_011',
             'clef_133', 'a_32_095', 'clef_b_142', '1_004', 'bar_116', '16_025', 'chord_130', '8_b_r_018', 't24_189',
             'a_2_065', 'dot_b_150', '#_b_004', 'dot_148', '32_b_n_038', '#_3', 'a_4_073', '8_016', 'bar_122',
             '#_b_005', 'a_4_067', '16_b_r_032', 'chord_133', '#_1', 'bar_105', 't4_185', 'flat_153', '32_b_r_045',
             'a_4_070', 'natural_b_175', 'bar_118', 'dot_146', '#_b_006', 'flat_b_158', '16_b_n_028', 'bar_104',
             'natural_b_173', 'a_4_077', '1_002', '8_b_r_020', '#_b_008', '16_023', '32_b_r_048', 'a_4_075', '4_010',
             'bar_110', 't4_188', 'natural_b_172', 'natural_b_176', '16_b_n_026', 'bar_124', '32_b_n_040', 'p_177',
             't2_182', 'bar_b_126', 'p_178', 'a_1_053', 'flat_154', 'a_2_061', 'a_1_056', '2_005', '32_b_r_044',
             'bar_112', 't4_187', 'bar_120', 'bar_108', 'a_16_089', 'flat_b_157', 'clef_b_139', 'a_32_094', '16_021',
             'a_16_088', '1_003', 'a_8_078', 'bar_114', 'dot_b_153', 'clef_134', '8_b_n_018', 'a_16_094', 'flat_155',
             'p_181', 'bar_117', 'a_2_057', 'bar_102', 'bar_100', '16_b_n_030', '16_b_n_029', '#_b_003', 'a_16_093',
             'a_32_097', '32_b_n_043', '16_b_r_034', '4_012', 'a_1_054', 'chord_129', 'natural_b_171', 'bar_113',
             '4_009', 't44_b_197', 'chord_131', 'flat_b_159', 'a_16_090']

# 使用了 Keras 库中的 models 模块中的 load_model() 方法，将一个已经保存的训练好的模型加载到程序中
# 将一个名为 frame.h5 的模型文件从相对路径 resources/ 中加载到 pic_model 变量中。
# 这个模型文件的扩展名 .h5 表示它是使用 Keras 所支持的一种保存模型的格式，这种格式将完整的模型信息包含在一个二进制文件中，可以将一个模型和它的权重值、结构、运算方式等方面的所有信息全部保存到磁盘中。
# 在加载模型后，我们可以进一步使用这个模型对输入数据进行预测，或在一些特定场景下对其进行微调。
pic_model = tf.keras.models.load_model('resources/frame.h5')

# model = pickle.load(open('resources/nn_trained_model_hog.sav', 'rb'))

# 定义了一个名为 predict() 的函数，它接受一张图片路径作为参数，并使用先前加载的 pic_model 模型对这张图片进行分类预测
def predict(img_path):
    # 首先使用 OpenCV 库的 resize() 方法将输入的图片调整为大小为 100x100 的矩阵，并将其像素值缩放到 [0, 1] 的范围内
    features = cv2.resize(img_path, (100, 100)) / 255.0
    # 将 features 数组升维，从三维的矩阵 (height, width, channel) 升为四维的张量 (batch_size, height, width, channel)。
    # 在这里，我们增加了一个大小为 1 的 batch_size 维度和一个大小为 1 的额外通道维度。这是因为预处理后的图像将被输入到卷积神经网络模型中，而在输入时，我们需要将其转换成一个批量大小为 1 的 mini-batch。添加一个批处理的第一维度是为了与模型接口一致
    features = features[np.newaxis, :, :, np.newaxis]
    # 使用 concatenate() 方法，将三个相同的数组 features 沿着最后一个维度连接起来。最后一个维度 axis=-1 表示要将它们排列在一起。
    # 这一步的主要目的是增加输入图像的通道数，因为模型接受的输入必须具有三个通道。
    features = np.concatenate([features, features, features], -1)
    # 使用之前预加载好的 pic_model 模型进行预测。该函数通过调用 numpy() 将张量转换为 numpy 数组，并选择 numpy() 数组中的第一个元素。由于该模型已经经过训练，因此可以将其应用于输入，并返回通过网络获得的输出
    label_id = pic_model(features).numpy()[0]
    # 使用 argmax() 方法找到最大概率值所对应的输出标签的索引，然后使用该索引在 label_str 列表中查找标签字符串，并将其作为函数的输出返回
    label_id = np.argmax(label_id)
    return label_str[label_id]

# 定义recognize函数
def recognize(out_file, most_common, coord_imgs, imgs_with_staff, imgs_spacing, imgs_rows):
    black_names = ['4', '8', '8_b_n', '8_b_r', '16', '16_b_n', '16_b_r',
                   '32', '32_b_n', '32_b_r', 'a_4', 'a_8', 'a_16', 'a_32', 'chord']
    ring_names = ['2', 'a_2']
    whole_names = ['1', 'a_1']
    disk_size = most_common / 4
    if len(coord_imgs) > 1:
        out_file.write("{\n")
    for i, img in enumerate(coord_imgs):
        res = []
        prev = ''
        time_name = ''
        primitives, prim_with_staff, boundary = get_connected_components(
            img, imgs_with_staff[i])
        for j, prim in enumerate(primitives):
            prim = binary_opening(prim, square(
                np.abs(most_common - imgs_spacing[i])))
            saved_img = (255 * (1 - prim)).astype(np.uint8)
            labels = predict(saved_img)
            octave = None
            label = labels[0]
            if label in black_names:
                test_img = np.copy(prim_with_staff[j])
                test_img = binary_dilation(test_img, disk(disk_size))
                comps, comp_w_staff, bounds = get_connected_components(
                    test_img, prim_with_staff[j])
                comps, comp_w_staff, bounds = filter_beams(
                    comps, comp_w_staff, bounds)
                bounds = [np.array(bound) + disk_size - 2 for bound in bounds]

                if len(bounds) > 1 and label not in ['8_b_n', '8_b_r', '16_b_n', '16_b_r', '32_b_n', '32_b_r']:
                    l_res = []
                    bounds = sorted(bounds, key=lambda b: -b[2])
                    for k in range(len(bounds)):
                        idx, p = estim(
                            boundary[j][0] + bounds[k][2], i, imgs_spacing, imgs_rows)
                        l_res.append(f'{label_map[idx][p]}/4')
                        if k + 1 < len(bounds) and (bounds[k][2] - bounds[k + 1][2]) > 1.5 * imgs_spacing[i]:
                            idx, p = estim(
                                boundary[j][0] + bounds[k][2] - imgs_spacing[i] / 2, i, imgs_spacing, imgs_rows)
                            l_res.append(f'{label_map[idx][p]}/4')
                    res.append(sorted(l_res))
                else:
                    for bbox in bounds:
                        c = bbox[2] + boundary[j][0]
                        line_idx, p = estim(int(c), i, imgs_spacing, imgs_rows)
                        l = label_map[line_idx][p]
                        res.append(get_note_name(prev, l, label))
            elif label in ring_names:
                head_img = 1 - binary_fill_holes(1 - prim)
                head_img = binary_closing(head_img, disk(disk_size))
                comps, comp_w_staff, bounds = get_connected_components(
                    head_img, prim_with_staff[j])
                for bbox in bounds:
                    c = bbox[2] + boundary[j][0]
                    line_idx, p = estim(int(c), i, imgs_spacing, imgs_rows)
                    l = label_map[line_idx][p]
                    res.append(get_note_name(prev, l, label))
            elif label in whole_names:
                c = boundary[j][2]
                line_idx, p = estim(int(c), i, imgs_spacing, imgs_rows)
                l = label_map[line_idx][p]
                res.append(get_note_name(prev, l, label))
            elif label in ['bar', 'bar_b', 'clef', 'clef_b', 'natural', 'natural_b', 't24', 't24_b', 't44',
                           't44_b'] or label in []:
                continue
            elif label in ['#', '#_b']:
                if prim.shape[0] == prim.shape[1]:
                    prev = '##'
                else:
                    prev = '#'
            elif label in ['cross']:
                prev = '##'
            elif label in ['flat', 'flat_b']:
                if prim.shape[1] >= 0.5 * prim.shape[0]:
                    prev = '&&'
                else:
                    prev = '&'
            elif label in ['dot', 'dot_b', 'p']:
                if len(res) == 0 or (
                        len(res) > 0 and res[-1] in ['flat', 'flat_b', 'cross', '#', '#_b', 't24', 't24_b', 't44',
                                                     't44_b']):
                    continue
                res[-1] += '.'
            elif label in ['t2', 't4']:
                time_name += label[1]
            elif label == 'chord':
                img = thin(1 - prim.copy(), max_iter=20)
                head_img = binary_closing(1 - img, disk(disk_size))
            if label not in ['flat', 'flat_b', 'cross', '#', '#_b']:
                prev = ''
        if len(time_name) == 2:
            out_file.write("[ " + "\\" + "meter<\"" + str(time_name[0]) + "/" + str(time_name[1]) + "\">" + ' '.join(
                [str(elem) if type(elem) != list else get_chord_notation(elem) for elem in res]) + "]\n")
        elif len(time_name) == 1:
            out_file.write("[ " + "\\" + "meter<\"" + '4' + "/" + '2' + "\">" + ' '.join(
                [str(elem) if type(elem) != list else get_chord_notation(elem) for elem in res]) + "]\n")
        else:
            out_file.write("[ " + ' '.join(
                [str(elem) if type(elem) != list else get_chord_notation(elem) for elem in res]) + "]\n")

    if len(coord_imgs) > 1:
        out_file.write("}")
    print("###########################", res, "##########################")


def start_server():
    app = make_handle_data()
    CORS(app, resources=r'/*')

    http_server = WSGIServer(('0.0.0.0', port), app)
    http_server.start()

    print(f"serving start on port {port}")
    # return app
    return http_server


def model_result(img_path):
    img = io.imread(img_path)
    img = gray_img(img)
    horizontal = IsHorizontal(img)
    if horizontal == False:
        theta = deskew(img)
        img = rotation(img, theta)
        img = get_gray(img)
        img = get_thresholded(img, threshold_otsu(img))
        img = get_closer(img)
        horizontal = IsHorizontal(img)

    original = img.copy()
    gray = get_gray(img)
    bin_img = get_thresholded(gray, threshold_otsu(gray))

    segmenter = Segmenter(bin_img)
    imgs_with_staff = segmenter.regions_with_staff
    most_common = segmenter.most_common

    imgs_spacing = []
    imgs_rows = []
    coord_imgs = []
    for i, img in enumerate(imgs_with_staff):
        spacing, rows, no_staff_img = coordinator(img, horizontal)
        imgs_rows.append(rows)
        imgs_spacing.append(spacing)
        coord_imgs.append(no_staff_img)

    print("Recognize...")
    out_file = open(f'static/new.txt', "w")
    recognize(out_file, most_common, coord_imgs,
              imgs_with_staff, imgs_spacing, imgs_rows)
    out_file.close()

    with open(f'static/new.txt', 'r') as f:
        line = f.readlines()[0]
    line = line.replace('[', '').replace(']', '').replace('\n', '').replace('{', '').replace('}', '').replace('#',
                                                                                                              '').replace(
        '&', '')
    line = line.split()
    mid = MidiFile()  # 给自己的文件定的.mid后缀
    track = MidiTrack()  # 定义声部，一个MidoTrack()就是一个声部
    mid.tracks.append(track)  # 这一句一定要加，表示将这个轨道加入到文件中，否则打不开（后面的play）
    track.append(Message('program_change', channel=0, program=0, time=0))
    for l in line:
        l = l.split('/')
        print(l[0])
        note = librosa.note_to_midi(l[0]) + 21
        track.append(
            Message('note_on', note=note, velocity=64, time=1000, channel=0))
        track.append(Message('note_off', note=note, velocity=64, time=1000, channel=0))

    mid.save(f'static/new.mid')
    midi_data = converter.parse("static/new.mid")

    # Convert MIDI data to MusicXML format
    xml_data = midi_data[0].write('musicxml')
    shutil.move(xml_data, 'static/new.musicxml')
    wav_path = f'new_{np.random.randint(1, 10000)}.wav'
    sy = FluidSynth(sound_font='resources/GS.sf2', sample_rate=16000)
    sy.midi_to_audio('static/new.mid', f'static/{wav_path}')
    return wav_path


def make_handle_data():
    app = Flask(__name__, template_folder="static", static_folder="static", static_url_path="/")
    CORS(app)

    @app.route('/')
    def home():
        return render_template("login.html")

    @app.route('/recognition', methods=['POST', 'OPTIONS'])
    @cross_origin()
    def upload_audio():
        format_time = time.strftime("%m_%d_%H_%M_%S", time.localtime(time.time()))

        upload_path = 'resources'
        os.makedirs(upload_path, exist_ok=True)

        # 存储音频
        img_files = []
        index = 1
        upload_files = request.files.to_dict()
        for file_name in upload_files:
            file = upload_files[file_name]
            img_path = f'{upload_path}/{format_time}_{index}.jpg'
            file.save(img_path)
            img_files.append(img_path)
            index = index + 1

        return jsonify({'result': model_result(img_files[0])})

    @app.route('/download_audio', methods=['POST', 'GET'])
    def download_audio():
        return app.send_static_file('new.musicxml')

    @app.route('/login', methods=['POST', 'OPTIONS'])
    @cross_origin()
    def login():
        # 获取value值
        username = request.json['username']
        password = request.json['password']

        with open('static/user.json', 'r') as f:
            user_info = json.load(f)

        is_match = 0
        for users in user_info['userinfo']:
            if users[0] == username and users[1] == password:
                is_match = 1
                break

        return jsonify({'result': is_match})

    @app.route('/register', methods=['POST', 'OPTIONS'])
    def register():
        # 获取value值
        username = request.json['username']
        password = request.json['password']

        with open('static/user.json', 'r') as f:
            user_info = json.load(f)

        users = user_info['userinfo']
        users.append([username, password])

        with open('static/user.json', 'w') as f:
            json.dump(user_info, f)

        return jsonify({'result': 'OK'})

    return app


if __name__ == '__main__':
    port = 8000
    http_server = start_server()
    try:
        http_server._stop_event.wait()
    finally:
        Greenlet.spawn(http_server.stop, timeout=http_server.stop_timeout).join()
