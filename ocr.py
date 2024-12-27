from paddleocr import PaddleOCR, draw_ocr
import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import os

# 初始化两个OCR对象
# 用于识别手写数字
hand_write_digit_ocr = PaddleOCR(
    use_angle_cls=True, 
    lang='en',  # 英文模型更适合识别数字
    rec=True,
    rec_model_dir='./ch_PP-OCRv4_rec_hand_infer/',
    rec_algorithm='SVTR_LCNet',
    max_text_length=3,  # 限制最大长度为3（最大2位数）
    # 方案1：使用内置的数字字典
    # rec_char_dict_path='ppocr/utils/dict/en_dict.txt',  # 使用英文字典，包含数字
    # 或者方案2：使用绝对路径
    # rec_char_dict_path=os.path.abspath('./ppocr/utils/dict/digit_dict.txt'),
    drop_score=0.3  # 降低阈值以提高召回率
)

# 用于识别表格文本
table_ocr = PaddleOCR(
    use_angle_cls=True,
    lang='ch',  # 中文模型
    rec=True,
    rec_model_dir='./ch_PP-OCRv4_rec_infer/',  # 使用标准模型
    rec_algorithm='SVTR_LCNet',
    drop_score=0.5
)

def convert_to_cv2_image(file_path):
    """将不同格式的文件转换为OpenCV图像格式"""
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.pdf':
        # 转换PDF的第一页为图像
        pages = convert_from_path(file_path, dpi=300)  # 使用较高的DPI以保证质量
        page = pages[0]  # 获取第一页
        # 转换为OpenCV格式
        open_cv_image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
        return open_cv_image
        
    elif file_ext in ['.jpg', '.jpeg', '.png']:
        # 直接读取图像文件
        return cv2.imread(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {file_ext}")

def process_file(file_path):
    """处理文件（支持PDF和图像格式）"""
    try:
        # 转换文件为OpenCV图像格式
        image = convert_to_cv2_image(file_path)
        if image is None:
            raise ValueError(f"无法读取文件: {file_path}")
        
        # 提取评分列
        score_column = extract_score_column(image)
        name_column = extract_name_column(image)
        cv2.imwrite('./output/score_column.jpg', score_column)
        cv2.imwrite('./output/name_column.jpg', name_column)

        # 识别评分列（手写数字）
        score_results = hand_write_digit_ocr.ocr(score_column, cls=True)
        
        # 识别整个表格（文本内容）
        table_results = table_ocr.ocr(name_column, cls=True)
        
        # name_results = table_ocr.ocr(name_column, cls=True)
        # name_data = []
        # print('识别名称结果')
        # if name_results:
        #     for line in name_results:
        #         if line:
        #             for item in line:
        #                 text = item[1][0]
        #                 confidence = item[1][1]
        #                 print(text, confidence)
        #                 if confidence > 0.5:
        #                     name_data.append(text)
        # print(name_data)

        # 处理表格文本结果
        table_data = []
        if table_results:
            for line in table_results:
                if line:
                    row_data = []
                    for item in line:
                        text = item[1][0]
                        confidence = item[1][1]
                        if confidence > 0.5:
                            row_data.append(text)
                    if row_data:
                        table_data.append(row_data)
        
        # 处理评分结果
        scores = []
        score_confidences = []  # 添加置信度列表
        if score_results:
            for line in score_results:
                if line:
                    for item in line:
                        text = item[1][0]
                        confidence = item[1][1]
                        # 打印原始识别结果和置信度
                        print(f"原始识别文本: {text}, 置信度: {confidence:.3f}")
                        
                        # 只保留数字
                        text = ''.join(c for c in text if c.isdigit())
                        if text :
                            try:
                                score = int(text)
                                if 0 <= score <= 100:
                                    scores.append(str(score))
                                    score_confidences.append(confidence)  # 保存对应的置信度
                            except:
                                continue
        
        # 打印调试信息
        print("表格文本识别结果:")
        for row in table_data:
            print(row)
        print("\n评分识别结果:")
        for score, conf in zip(scores, score_confidences):
            print(f"分数: {score}, 置信度: {conf:.3f}")
        
        return table_data, scores, score_confidences  # 返回置信度
        
    except Exception as e:
        print(f"处理文件时出错: {str(e)}")
        return None, None, None

def process_multi_page_pdf(pdf_path):
    """处理多页PDF文件"""
    try:
        # 转换PDF的所有页面
        pages = convert_from_path(pdf_path, dpi=300)
        
        all_table_data = []
        all_scores = []
        all_confidences = []
        
        # 处理每一页
        for i, page in enumerate(pages):
            # 转换为OpenCV格式
            image = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
            
            # 保存中间结果用于调试
            cv2.imwrite(f'./output/page_{i+1}.jpg', image)
            
            # 处理单页
            table_data, scores, confidences = process_single_page(image)
            
            if table_data:
                all_table_data.extend(table_data)
                all_scores.extend(scores)
                all_confidences.extend(confidences)
        
        return all_table_data, all_scores, all_confidences
        
    except Exception as e:
        print(f"处理PDF文件时出错: {str(e)}")
        return None, None, None

def process_single_page(image):
    """处理单个图像页面"""
    # 提取评分列
    score_column = extract_score_column(image)
    name_column = extract_name_column(image)
    print(len(name_column))
    # cv2.imwrite('score_column.jpg', score_column)
    # cv2.imwrite('name_column.jpg', name_column)
    # 识别评分列（手写数字）
    # score_results = hand_write_digit_ocr.ocr(score_column, cls=True)
     
    table_data = []
    name_idx = 0
    # 识别评分人和单位
    for name_img in name_column:
        cv2.imwrite(f'./output/name_column_{name_idx}.jpg', name_img)
        name_idx += 1
        table_results = table_ocr.ocr(name_img, cls=True)
        if table_results:
            for line in table_results:
                if line:
                    row_data = []
                    for item in line:
                        text = item[1][0]
                        confidence = item[1][1]
                        if confidence > 0.5:
                            row_data.append(text)
                    if row_data:
                        table_data.append(row_data)
    
    # 处理评分结果
    scores = []
    score_confidences = []  # 添加置信度列表
    score_idx = 0
    print(len(score_column))
    for score_img in score_column:
        # processed_img = preprocess_score_image(score_img)
        processed_img = score_img
        cv2.imwrite(f'./output/score_column_process_{score_idx}.jpg', processed_img)
        score_idx += 1
        # assert processed_img is not None
        score_results = hand_write_digit_ocr.ocr(processed_img, cls=True) 
        if score_results:
            for line in score_results:
                if line:
                    for item in line:
                        text = item[1][0]
                        confidence = item[1][1]
                        print(f'score_results: {text}, confidence is: {confidence}')
                        # 后处理识别结果
                        processed_score, processed_confidence = post_process_score(text, confidence)
                        if processed_score:
                            scores.append(processed_score)
                            score_confidences.append(processed_confidence)
        else:
            print('score_results is None')
    
    
    # 打印调试信息
    print("表格文本识别结果:")
    for row in table_data:
        print(row)
    print("\n评分识别结果:")
    for score, conf in zip(scores, score_confidences):
        print(f"分数: {score}, 置信度: {conf:.3f}")
    print('debug',len(table_data), len(scores))
    return table_data, scores, score_confidences  # 返回置信度

def merge_nearby_lines(coordinates, threshold=20):
    """合并相近的线条坐标
    
    Args:
        coordinates: 线条坐标列表
        threshold: 合并阈值，如果两条线的距离小于此值则合并
    
    Returns:
        合并后的线条坐标列表
    """
    if not coordinates:
        return []
    
    # 确保坐标是排序的
    sorted_coords = sorted(coordinates)
    merged = []
    current_group = [sorted_coords[0]]
    
    # 遍历所有坐标
    for coord in sorted_coords[1:]:
        # 如果当前坐标与组内最后一个坐标的距离小于阈值
        if coord - current_group[-1] < threshold:
            current_group.append(coord)
        else:
            # 将当前组的平均值添加到结果中
            merged.append(sum(current_group) // len(current_group))
            current_group = [coord]
    
    # 处理最后一组
    merged.append(sum(current_group) // len(current_group))
    
    return merged

def detect_vertical_lines(image):
    """检测图像中的垂直线条"""
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 二值化
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # 定义垂直线检测核
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 205))
    
    # 检测垂直线
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    print('how many vertical lines:', len(vertical_lines))

    # 查找轮廓
    contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(">>>>>>>>>>>>")
    # 获取垂直线的x坐标
    x_coordinates = []
    for contour in contours:
        x, _, w, _ = cv2.boundingRect(contour)
        # print(x, w)
        x_coordinates.append(x + w//2)
    print('检测到的原始垂直线数量:', len(x_coordinates))
    
    # 合并相近的线条
    merged_coordinates = merge_nearby_lines(x_coordinates, threshold=20)
    print('合并后的垂直线数量:', len(merged_coordinates))
    print('合并后的垂直线位置:', merged_coordinates)
    print(">>>>>>>>>>>>")

    return merged_coordinates

def detect_horizontal_lines(image):
    """检测图像中的水平线条"""
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 二值化
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # 定义水平线检测核
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (205, 1))
    
    # 检测水平线
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    print('水平线图像大小:', horizontal_lines.shape)

    # 查找轮廓
    contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 获取水平线的y坐标
    y_coordinates = []
    for contour in contours:
        _, y, _, h = cv2.boundingRect(contour)
        print(f'水平线位置: y={y}, 高度={h}')
        y_coordinates.append(y + h//2)
    
    print('检测到的原始水平线数量:', len(y_coordinates))
    
    # 合并相近的水平线
    merged_coordinates = merge_nearby_lines(y_coordinates, threshold=15)  # 水平线可以用更小的阈值
    print('合并后的水平线数量:', len(merged_coordinates))
    print('合并后的水平线位置:', merged_coordinates)
    
    return merged_coordinates  # 返回排序后的y坐标列表

def draw_all_lines(image, vertical_lines, horizontal_lines, filename='./output/debug_all_lines.jpg'):
    """在图像上画出所有检测到的水平线和垂直线"""
    debug_image = image.copy()
    
    # 画出所有垂直线（绿色）
    for x in vertical_lines:
        cv2.line(debug_image, (x, 0), (x, image.shape[0]), (0, 255, 0), 2)
    
    # 画出所有水平线（蓝色）
    for y in horizontal_lines:
        cv2.line(debug_image, (0, y), (image.shape[1], y), (255, 0, 0), 2)
    
    cv2.imwrite(filename, debug_image)
    return debug_image

def extract_name_column(image):
    """提取名称列（第4列）"""
    # 获取垂直线和水平线位置
    vertical_lines = detect_vertical_lines(image)
    horizontal_lines = detect_horizontal_lines(image)
    print(vertical_lines)
    print(horizontal_lines)
    
    # 画出所有检测到的线
    draw_all_lines(image, vertical_lines, horizontal_lines, './output/debug_all_lines.jpg')
    
    if len(vertical_lines) < 5:  # 确保至少检测到5条垂直线
        print("警告：未能检测到足够的垂直线，使用备用方法")
        # 使用原来的备用方法
        print("使用原来的备用方法")
        height, width = image.shape[:2]
        column_widths = [width * ratio for ratio in [0.1, 0.15, 0.25, 0.15, 0.15]]
        name_x = int(sum(column_widths[:3]))
        name_w = int(column_widths[3])
    else:
        # 使用检测到的垂直线确定姓名的位置
        # 姓是第2,3列，以使用第2和第4条垂直线作为边界
        name_x = vertical_lines[1]
        name_w = vertical_lines[3] - vertical_lines[1]
    first_horizontal_line = horizontal_lines[0]
    # 提取评分列
    name_column = []
    for i in range(len(horizontal_lines) - 1 ):
        name_column.append(image[horizontal_lines[i]:horizontal_lines[i+1], name_x:name_x+name_w])
    # name_column.append(image[first_horizontal_line:, name_x:name_x+name_w])
    print(name_x, name_w)
    # 保存调试图像
    debug_image = image.copy()
    # 画垂直线
    cv2.line(debug_image, (name_x, 0), (name_x, image.shape[0]), (0, 255, 0), 2)
    cv2.line(debug_image, (name_x+name_w, 0), (name_x+name_w, image.shape[0]), (0, 255, 0), 2)
    # 画水平线
    cv2.line(debug_image, (0, first_horizontal_line), (image.shape[1], first_horizontal_line), (255, 0, 0), 2)
    
    cv2.imwrite('./output/debug_name_column.jpg', debug_image)
    
    return name_column

def extract_score_column(image):
    """提取评分列（第4列）"""
    # 获取垂直线位置
    vertical_lines = detect_vertical_lines(image)
    horizontal_lines = detect_horizontal_lines(image)
    print(vertical_lines)
    print(horizontal_lines)
    if len(vertical_lines) < 5:  # 确保至少检测到5条直线
        print("警告：未能检测到足够的垂直线，使用备用方法")
        # 使用原来的备用方法
        height, width = image.shape[:2]
        column_widths = [width * ratio for ratio in [0.1, 0.15, 0.25, 0.15, 0.15]]
        score_x = int(sum(column_widths[:3]))
        score_w = int(column_widths[3])
    else:
        # 使用检测到的垂直线确定评分列的位置
        # 评分列是第4列，所以使用第3和第4条垂直线作为边界
        score_x = vertical_lines[3]
        score_w = vertical_lines[4] - vertical_lines[3]
    first_horizontal_line = horizontal_lines[1]
    # 提取评分列
    score_column = []
    for i in range(1, len(horizontal_lines) - 1 ):
        score_column.append(image[horizontal_lines[i]:horizontal_lines[i+1], score_x:score_x+score_w])
    # score_column.append(image[first_horizontal_line:, score_x:score_x+score_w])
    print(score_x, score_w)
    # 保存调试图像
    debug_image = image.copy()
    cv2.line(debug_image, (score_x, 0), (score_x, image.shape[0]), (0, 255, 0), 2)
    cv2.line(debug_image, (score_x+score_w, 0), (score_x+score_w, image.shape[0]), (0, 255, 0), 2)
    cv2.imwrite('./output/debug_score_column.jpg', debug_image)
    
    return score_column

def preprocess_score_image(score_img):
    """预处理分数图像"""
    # 转换为灰度图
    gray = cv2.cvtColor(score_img, cv2.COLOR_BGR2GRAY)
    
    # 调整图像大小，保持一致的尺寸
    height = 32  # PaddleOCR推荐的高度
    ratio = height / gray.shape[0]
    width = int(gray.shape[1] * ratio)
    resized = cv2.resize(gray, (width, height))
    
    # 使用OTSU自适应阈值进行二值化
    _, binary = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 轻微的模糊处理，去除噪点
    denoised = cv2.GaussianBlur(binary, (3,3), 0)
    
    # 轻微的膨胀，使数字更清晰
    kernel = np.ones((2,2), np.uint8)
    dilated = cv2.dilate(denoised, kernel, iterations=1)
    
    # 转回三通道图像（PaddleOCR需要）
    result = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
    
    return result

def post_process_score(text, confidence):
    """后处理识别结果"""
    # 只保留数字
    digits = ''.join(c for c in text if c.isdigit())
    print(f'before mapping digits: {digits} and text is {text}')
    # 如果识别结果包含字母，尝试转换常见的误识别情况
    mapping = {
        'O': '0',
        'I': '1',
        'l': '1',
        'B': '8',
        'S': '5',
        'Z': '2',
        'q': '9',
        'b': '6',
        'Q': '9',
         
    }
    digits = ''
    for char in text:
        if char in mapping:
            digits += mapping[char]
        elif char.isdigit():
            digits += char
    if len(digits) >= 3 : 
        digits = digits[-2:]
    print(f'after mapping digits: {digits} and text is {text}')
    
    score = int(digits)
        
    return str(score), confidence
    

if __name__ == "__main__":
    
    # file_path = sys.argv[1]
    file_path = './test_data/test2.pdf'
    file_ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_ext == '.pdf':
            # 处理PDF文件
            table_data, scores, confidences = process_multi_page_pdf(file_path)
        else:
            # 处理图像文件
            table_data, scores, confidences = process_file(file_path)
        # print(len(table_data), len(scores))
        print(table_data)
        result = {}
        score_idx = 0
        names = []
        score_data = []
        while score_idx < len(scores):
                score_data.append([scores[score_idx], confidences[score_idx]])
                score_idx += 1
        result['title'] = table_data[0]
        result['names'] = table_data[1:]
        result['score'] = score_data
        import json
        print(json.dumps(result,ensure_ascii=False))

                    
    except Exception as e:
        print(f"处理文件时出错: {str(e)}")