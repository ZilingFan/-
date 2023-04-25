# -*- coding: utf-8 -*-
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re

from PIL import Image
import time
import fitz
import numpy as np
import pandas as pd
import copy
import cv2
import os
import sys
import subprocess


__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))


from core.tools.infer.utility import draw_ocr_box_txt
from core.ppocr.utils.logging import get_logger
from core.ppocr.utils.utility import get_image_file_list, check_and_read_gif
from core.utils.pdf_utils import pdf2image
import core.tools.infer.predict_cls as predict_cls
import core.tools.infer.predict_det as predict_det
import core.tools.infer.predict_rec as predict_rec
import core.tools.infer.utility as utility


os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

logger = get_logger()


class TextSystem(object):
    def __init__(self, args):
        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]])
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        dst_img_height, dst_img_width = dst_img.shape[0:2]
        if dst_img_height * 1.0 / dst_img_width >= 1.5:
            dst_img = np.rot90(dst_img)
        return dst_img

    def print_draw_crop_rec_res(self, img_crop_list, rec_res):
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite("./output/img_crop_%d.jpg" % bno, img_crop_list[bno])
            logger.info(bno, rec_res[bno])

    def __call__(self, img):
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        logger.info("dt_boxes num : {}, elapse : {}".format(
            len(dt_boxes), elapse))
        if dt_boxes is None:
            return None, None
        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = self.get_rotate_crop_image(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            logger.info("cls num  : {}, elapse : {}".format(
                len(img_crop_list), elapse))

        rec_res, elapse = self.text_recognizer(img_crop_list)
        logger.info("rec_res num  : {}, elapse : {}".format(
            len(rec_res), elapse))
        # self.print_draw_crop_rec_res(img_crop_list, rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_reuslt in zip(dt_boxes, rec_res):
            text, score = rec_reuslt
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_reuslt)
        return filter_boxes, filter_rec_res


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
                (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes

# 将confident列中按值分类


def highlight_confident(val):
    color = ''
    if val <= 0.80:
        color = 'background-color:red'
    elif val <= 0.90:
        color = 'background-color:yellow'
    return color


def decoder(input, map_dict):
    print(input, " -----------------------")
    values = input.replace(",", ".").strip("\n").split(' ')
    tmp_vlues = values.copy()
    res = {}
    for i in tmp_vlues:
        key = ""
        for k, v in map_dict.items():
            if k in i:
                values.remove(i)
                key = v
                break
        """
        if "//" in i:
            values.remove(i)
            key = "双斜杠"
            break
        elif "①" in i:
            values.remove(i)
            key = "平面图"
            break
        """
    res["key"] = key
    res["value"] = ' '.join(values)
    return res


def list_image_input_file(dir_path):
    """将某个文件夹内所有以input.png结尾的图片文件的详细路径筛选出来

    Args:
        dir_path (str): 要筛选的具体文件夹

    Returns:
        input_images (list): 一个list，里面包含具体路径
    """
    input_images = []
    files = os.listdir(dir_path)
    for filename in files:
        if filename[-9: ] == 'input.png':
            input_images.append(os.path.join(dir_path, filename))
            
    return input_images
    

def main(args, max_pdf_pixel):
    main_start = time.time()
    # 读取映射字典
    map_dict = {}
    with open(os.path.join(__dir__, 'map_dict.txt'), 'r', encoding='utf-8') as f:
        data = f.readlines()
        for d in data:
            k, v = d.split()
            print(k, v)
            map_dict[k] = v

    draw_img_save = args.output_dir
    
    img_files = list_image_input_file(args.image_dir)
    image_file_list = get_image_file_list(img_files)
    # image_file_list = image_file_list[args.process_id::args.total_process_num]
    text_sys = TextSystem(args)
    is_visualize = True
    font_path = args.vis_font_path
    drop_score = args.drop_score
    for image_file in image_file_list:
        img, flag = check_and_read_gif(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        starttime = time.time()

        # 调整一次照片尺寸
        h, w, _ = img.shape
        l = max(h, w)
        if l > max_pdf_pixel:
            zoom = max_pdf_pixel / l
            img = cv2.resize(img, dsize=(0, 0), fx=zoom, fy=zoom)

        dt_boxes, rec_res = text_sys(img)
        # 解析识别结果，将一些特殊字符转化为语义信息
        for i in range(len(rec_res)):
            res = rec_res[i][0]
            for k, v in map_dict.items():
                res = res.replace(k, v)
            rec_res[i] = (res, rec_res[i][1])

        elapse = time.time() - starttime
        logger.info("Predict time of %s: %.3fs" % (image_file, elapse))

        for text, score in rec_res:
            logger.info("{}, {:.3f}".format(text, score))

        if is_visualize:
            image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            boxes = dt_boxes
            txts = [rec_res[i][0] for i in range(len(rec_res))]
            scores = [rec_res[i][1] for i in range(len(rec_res))]

            """
            # print(type(image))
            print(image)
            # print(type(boxes))
            print(boxes)
            # print(type(txts))
            print(txts)
            # print(type(scores))
            print(scores)
            """

            # 这部分先注释掉，也会花很多时间
            
            draw_img = draw_ocr_box_txt(
                image,
                boxes,
                txts,
                scores,
                drop_score=drop_score,
                font_path=font_path)

            # if not os.path.exists(draw_img_save):
            #     os.makedirs(draw_img_save)
            fn = os.path.basename(image_file)[:-10]
            tmp = os.path.join(draw_img_save, f'{fn}_out.png')
            cv2.imwrite(tmp, draw_img[:, :, ::-1])
            # logger.info("The visualized image saved in {}".format(
            #     os.path.join(draw_img_save, os.path.basename(image_file))))

            # with open(draw_img_save+fn+".txt","w", encoding='utf8') as file:
            #     for id in rec_res:
            #         file.write(str(id[0])+", "+str(id[1])+"\n")
            data = list()

            
            # 语义解析
            for id in rec_res:
                semantics = decoder(id[0], map_dict)
                data.append((semantics['key'], semantics['value'], id[1]))


            df = pd.DataFrame(data=rec_res, columns=['value', "confidence"])

            df[['下公差限', '名义值', '上公差限']] = pd.DataFrame(df['value'].apply(gongcha_analyse).values.tolist())
            # for id in values:
            #     print(id[0], id[1], "---------------------------------------")
            values=df['value'].apply(gongcha_analyse).values.tolist()
            lower=[values[i][0] for i in range(len(values))]
            notion=[values[i][1] for i in range(len(values))]
            upper=[values[i][2] for i in range(len(values))]

            # 根据confident设置表格style
            df = df.style.applymap(highlight_confident, subset=["confidence"])

            df.to_excel(os.path.join(draw_img_save, f'{fn}_out.xlsx'), sheet_name='Sheet1')

            with open(os.path.join(draw_img_save, f'{fn}_out.html'), 'w') as f:
                f.write(df.set_table_attributes(
                    'border="1" class="dataframe table table-hover table-bordered"').set_precision(3).render())
            # df.to_html(draw_img_save+'red_left.html',index=True,border=5,justify='left')  # 无行索引
            
            print('recognize cost {:.5f}(s)'.format(time.time() - main_start))
    return txts, boxes, scores, lower, notion, upper, data

def gongcha_analyse(result):
    # 设置一个flag如果为True表示正确解析除了一个公差
    f = True

    if '±' in result:
        split_list = result.split('±')

        if len(split_list) != 2:
            return '', '', ''
        else:
            value, bias = split_list[0], split_list[1]
            return value + '-' + bias, value, value + '+' + bias 
    else:
        return '', '', ''


def predict_system(file_dir, file_name, is_pdf= True, zoom_plus=7, max_pdf_pixel=3500):
    load_start = time.time()
    """封装识别接口

    Args:
        file_dir (str): 文件夹位置
        file_name (str): 文件名，不含后缀名

    Returns:
        识别结果 (dict): 最后的识别结果，组织在一个dict内
    """
    args = utility.parse_args()[0]

    # if is_pdf:
    #     zoom_plus = zoom_plus
    #     rot_ang = 0
    #     pdf2image(file_dir, file_name, zoom_plus, zoom_plus, rot_ang)
    
    # 设置参数值
    args.image_dir = file_dir
    args.output_dir = file_dir
    print('load cost {}(s)'.format(time.time() - load_start))

    if args.use_mp:
        p_list = []
        total_process_num = args.total_process_num
        for process_id in range(total_process_num):
            cmd = [sys.executable, "-u"] + sys.argv + [
                "--process_id={}".format(process_id),
                "--use_mp={}".format(False)
            ]
            p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout)
            p_list.append(p)
        for p in p_list:
            p.wait()
    else:
        return main(args, max_pdf_pixel)


if __name__ == "__main__":
    args = utility.parse_args()[0]

    if args.image_dir[-3:] == "pdf":
        '''
        # 将PDF转化为图片
        pdfPath pdf文件的路径
        imgPath 图像要保存的文件夹
        zoom_x x方向的缩放系数
        zoom_y y方向的缩放系数
        rotation_angle 旋转角度
        '''
        def pdf_image(pdfPath, imgPath, zoom_x, zoom_y, rotation_angle):
            # 打开PDF文件
            pdf = fitz.open(pdfPath)
            # 逐页读取PDF
            for pg in range(0, pdf.pageCount):
                page = pdf[pg]
                # 设置缩放和旋转系数
                trans = fitz.Matrix(zoom_x, zoom_y).preRotate(rotation_angle)
                pm = page.getPixmap(matrix=trans, alpha=False)
                # 开始写图像
                pm.writePNG(imgPath+"red_left.png")
                args.image_dir = imgPath+"red_left.png"
            pdf.close()

        zoom_plus = 10
        rot_ang = 0
        pdf_image(args.image_dir, args.output_dir,
                  zoom_plus, zoom_plus, rot_ang)

    if args.use_mp:
        p_list = []
        total_process_num = args.total_process_num
        for process_id in range(total_process_num):
            cmd = [sys.executable, "-u"] + sys.argv + [
                "--process_id={}".format(process_id),
                "--use_mp={}".format(False)
            ]
            p = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stdout)
            p_list.append(p)
        for p in p_list:
            p.wait()
    else:
        main(args)
