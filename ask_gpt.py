from paddleocr import PaddleOCR, draw_ocr
import cv2
import numpy as np

# 初始化两个OCR对象
# 用于识别手写数字
hand_write_digit_ocr = PaddleOCR(
    use_angle_cls=True, 
    lang='en',  # 英文模型更适合识别数字
    rec=True,
    rec_model_dir='./ch_PP-OCRv4_rec_hand_infer/',
    rec_algorithm='SVTR_LCNet',
    max_text_length=3,  # 限制最大长度为3（最大2位数）
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
# def detect_vertical_lines(image):
#     """检测图像中的垂直线条"""
#     # 转换为灰度图
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     # 二值化，尝试不同的阈值
#     _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
#     # 定义垂直线检测核
#     vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 105))
    
#     # 先进行闭运算，再进行开运算
#     closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, vertical_kernel)
#     vertical_lines = cv2.morphologyEx(closed, cv2.MORPH_OPEN, vertical_kernel)

#     # 查找轮廓
#     contours, _ = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     # 获取垂直线的x坐标
#     x_coordinates = []
#     for contour in contours:
#         x, _, w, h = cv2.boundingRect(contour)
#         # 过滤小轮廓
#         if w > 10:  # 只保留宽度大于10的轮廓
#             x_coordinates.append(x + w // 2)
#     print(len(x_coordinates))
#     return sorted(set(x_coordinates))  # 使用 set 去重并排序

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
        print(x, w)
        x_coordinates.append(x + w//2)
    print('how many vertical lines:', len(x_coordinates))
    print(">>>>>>>>>>>>")

    return sorted(x_coordinates)

def extract_score_column(image):
    """提取评分列（第4列）"""
    # 获取垂直线位置
    vertical_lines = detect_vertical_lines(image)
    print(vertical_lines)
    if len(vertical_lines) < 5:  # 确保至少检测到5条垂直线
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
    
    # 提取评分列
    score_column = image[:, score_x:score_x+score_w]
    print(score_x, score_w)
    # 保存调试图像
    debug_image = image.copy()
    cv2.line(debug_image, (score_x, 0), (score_x, image.shape[0]), (0, 255, 0), 2)
    cv2.line(debug_image, (score_x+score_w, 0), (score_x+score_w, image.shape[0]), (0, 255, 0), 2)
    cv2.imwrite('debug_score_column.jpg', debug_image)
    
    return score_column

def process_table(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    
    # 提取评分列
    score_column = extract_score_column(image)
    cv2.imwrite('score_column.jpg', score_column)
    # 识别评分列（手写数字）
    score_results = hand_write_digit_ocr.ocr(score_column, cls=True)
    
    # 识别整个表格（文本内容）
    table_results = table_ocr.ocr(image, cls=True)
    
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
                    if text and confidence > 0.3:
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

if __name__ == "__main__":
    image_path = 'test.jpg'
    table_data, scores, confidences = process_table(image_path)
    
    # 合并结果
    print("\n最终结果:")
    score_idx = 0
    for row in table_data:
        if len(row) >= 3 and score_idx < len(scores):
            row_with_score = row[:3] + [
                f"{scores[score_idx]}(conf:{confidences[score_idx]:.3f})"
            ]
            print(row_with_score)
            score_idx += 1