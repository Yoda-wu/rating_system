import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from ocr import process_file, process_multi_page_pdf

app = Flask(__name__)

# 配置上传文件存储路径
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

# 确保上传目录存在
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/ocr", methods=['POST'])
def ocr_process():
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有文件上传'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
            
        if not allowed_file(file.filename):
            return jsonify({'error': '不支持的文件格式'}), 400
            
        # 安全地保存文件
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # 根据文件类型处理
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext == '.pdf':
            table_data, scores, confidences = process_multi_page_pdf(file_path)
        else:
            table_data, scores, confidences = process_file(file_path)
            
        if table_data is None:
            return jsonify({'error': '处理文件失败'}), 500
            
        # 整理返回数据
        result = {}
        result['name'] = [],
        result['score'] = []    
        score_idx = 0
        score_data = []
        while score_idx < len(scores):
                score_data.append([scores[score_idx], confidences[score_idx]])
                score_idx += 1
        result['title'] = table_data[0]
        result['names'] = table_data[1:]
        result['score'] = score_data
        
        # 清理上传的文件
        os.remove(file_path)
        
        return jsonify({
            'status': 0,
            'data': result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)