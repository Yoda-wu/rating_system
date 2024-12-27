import cv2
import numpy as np

def extract_score_row(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 使用自适应阈值进行二值化
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 遍历轮廓，寻找评分行
    for contour in contours:
        # 计算轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)
        print(x, y, w, h)
        # 根据 y 坐标判断是否为评分行（假设评分行在特定的 y 范围内）
        if y > 100 and y < 200:  # 根据实际情况调整 y 范围
            # 提取评分行图像
            score_row = image[y:y+h, x:x+w]
            return score_row

    return None

if __name__ == "__main__":
    image_path = "test.jpg"  # 替换为你的图像文件路径
    score_row_image = extract_score_row(image_path)
    
    if score_row_image is not None:
        cv2.imshow("Score Row", score_row_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # 保存评分行图像
        cv2.imwrite("score_row.jpg", score_row_image)
    else:
        print("未找到评分行。")