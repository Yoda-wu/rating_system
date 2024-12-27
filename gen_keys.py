def generate_dict():
    # 基础数字和符号
    numbers = [str(i) for i in range(10)]
    symbols = ['.', '%']
    
    # 生成字典内容
    chars = numbers + symbols
    
    # 写入文件
    with open('ppocr_keys_v1.txt', 'w', encoding='utf-8') as f:
        f.write(''.join(chars))

if __name__ == '__main__':
    generate_dict()

