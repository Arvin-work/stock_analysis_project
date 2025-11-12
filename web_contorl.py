from flask import Flask, render_template, request, jsonify, send_file
import subprocess
import threading
import os
import sys
import json
from datetime import datetime
import psutil
import GPUtil
import platform

app = Flask(__name__)

# 存储程序运行状态
program_status = {
    'is_running': False,
    'output': '',
    'start_time': None,
    'end_time': None
}

class ProgramRunner:
    def __init__(self):
        self.process = None
        self.output_lines = []
    
    def run_program(self, args=None):
        global program_status
        try:
            program_status['is_running'] = True
            program_status['start_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            program_status['output'] = ''
            self.output_lines = []
            
            # 构建命令
            cmd = [sys.executable, 'main.py']
            if args:
                cmd.extend(args)
            
            print(f"执行命令: {' '.join(cmd)}")
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                encoding='utf-8'
            )
            
            # 实时捕获输出
            while True:
                output = self.process.stdout.readline()
                if output == '' and self.process.poll() is not None:
                    break
                if output:
                    line = output.strip()
                    print(f"输出: {line}")
                    self.output_lines.append(line)
                    program_status['output'] = '\n'.join(self.output_lines[-100:])
            
            self.process.wait()
            
        except Exception as e:
            error_msg = f"错误: {str(e)}"
            print(error_msg)
            self.output_lines.append(error_msg)
            program_status['output'] = '\n'.join(self.output_lines)
        finally:
            program_status['is_running'] = False
            program_status['end_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def stop_program(self):
        if self.process:
            print("正在停止程序...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
                print("程序已正常停止")
            except subprocess.TimeoutExpired:
                print("程序未正常停止，强制终止")
                self.process.kill()
                self.process.wait()

def get_system_info():
    """获取系统信息"""
    try:
        # CPU信息
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        # 内存信息
        memory = psutil.virtual_memory()
        
        # 磁盘信息
        disk = psutil.disk_usage('.')
        
        # GPU信息 (如果有)
        gpus = []
        try:
            gpu_list = GPUtil.getGPUs()
            for gpu in gpu_list:
                gpus.append({
                    'name': gpu.name,
                    'load': gpu.load * 100,
                    'memory_used': gpu.memoryUsed,
                    'memory_total': gpu.memoryTotal,
                    'temperature': gpu.temperature
                })
        except:
            gpus = []
        
        # 系统信息
        system_info = {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'processor': platform.processor(),
            'cpu': {
                'percent': cpu_percent,
                'cores': cpu_count,
                'frequency': cpu_freq.current if cpu_freq else 'N/A'
            },
            'memory': {
                'percent': memory.percent,
                'used_gb': round(memory.used / (1024**3), 2),
                'total_gb': round(memory.total / (1024**3), 2)
            },
            'disk': {
                'percent': disk.percent,
                'used_gb': round(disk.used / (1024**3), 2),
                'total_gb': round(disk.total / (1024**3), 2)
            },
            'gpus': gpus
        }
        
        return system_info
    except Exception as e:
        print(f"获取系统信息错误: {e}")
        return {}

def get_file_tree(path='.'):
    """获取文件树结构 - 修复版本"""
    file_tree = []
    ignore_dirs = {'.git', '__pycache__', 'node_modules', '.vscode', '.idea', 'venv', 'env'}
    ignore_files = {'.DS_Store', 'Thumbs.db'}
    
    try:
        # 获取当前工作目录的绝对路径
        abs_path = os.path.abspath(path)
        
        for item in os.listdir(abs_path):
            if item in ignore_files:
                continue
                
            item_path = os.path.join(abs_path, item)
            relative_path = os.path.relpath(item_path, start='.')
            
            if os.path.isdir(item_path) and item not in ignore_dirs:
                try:
                    # 递归获取子目录，但限制深度避免性能问题
                    children = get_file_tree(item_path)
                    file_tree.append({
                        'name': item,
                        'type': 'directory',
                        'path': relative_path,
                        'children': children
                    })
                except PermissionError:
                    # 跳过无权限访问的目录
                    continue
            elif os.path.isfile(item_path):
                # 显示所有文件，不限制文件类型
                file_tree.append({
                    'name': item,
                    'type': 'file',
                    'path': relative_path,
                    'size': os.path.getsize(item_path)
                })
                
        # 按类型和名称排序：目录在前，文件在后
        file_tree.sort(key=lambda x: (x['type'] != 'directory', x['name'].lower()))
        
    except Exception as e:
        print(f"获取文件树错误: {e}")
        # 返回错误信息
        return [{'name': f'错误: {str(e)}', 'type': 'error', 'path': path}]
    
    return file_tree

def read_file_content(filepath):
    """读取文件内容"""
    try:
        # 安全检查：确保文件路径在当前目录下
        abs_path = os.path.abspath(filepath)
        current_dir = os.path.abspath('.')
        
        if not abs_path.startswith(current_dir):
            return "错误: 文件路径不安全"
            
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            with open(filepath, 'r', encoding='gbk') as f:
                return f.read()
        except:
            return "无法读取文件内容（编码问题）"
    except Exception as e:
        return f"读取文件错误: {str(e)}"

def save_file_content(filepath, content):
    """保存文件内容"""
    try:
        # 安全检查
        abs_path = os.path.abspath(filepath)
        current_dir = os.path.abspath('.')
        
        if not abs_path.startswith(current_dir):
            return False, "错误: 文件路径不安全"
            
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        return True, "文件保存成功"
    except Exception as e:
        return False, f"保存文件错误: {str(e)}"

runner = ProgramRunner()

@app.route('/')
def index():
    return render_template('ops_index.html')

@app.route('/api/status')
def get_status():
    return jsonify(program_status)

@app.route('/api/system')
def get_system_status():
    """获取系统状态"""
    system_info = get_system_info()
    return jsonify(system_info)

@app.route('/api/files/tree')
def get_files_tree():
    """获取文件树"""
    path = request.args.get('path', '.')
    file_tree = get_file_tree(path)
    return jsonify(file_tree)

@app.route('/api/files/content')
def get_file_content():
    """获取文件内容"""
    filepath = request.args.get('path')
    if not filepath:
        return jsonify({'error': '文件路径不能为空'}), 400
    
    # 安全检查
    abs_path = os.path.abspath(filepath)
    current_dir = os.path.abspath('.')
    
    if not abs_path.startswith(current_dir):
        return jsonify({'error': '文件路径不安全'}), 403
        
    if not os.path.exists(filepath):
        return jsonify({'error': '文件不存在'}), 404
    
    content = read_file_content(filepath)
    return jsonify({'content': content})

@app.route('/api/files/save', methods=['POST'])
def save_file():
    """保存文件"""
    data = request.get_json()
    if not data:
        return jsonify({'error': '请求数据为空'}), 400
        
    filepath = data.get('path')
    content = data.get('content')
    
    if not filepath:
        return jsonify({'error': '文件路径不能为空'}), 400
    
    success, message = save_file_content(filepath, content)
    if success:
        return jsonify({'message': message})
    else:
        return jsonify({'error': message}), 500

@app.route('/api/start', methods=['POST'])
def start_program():
    if program_status['is_running']:
        return jsonify({'status': 'error', 'message': '程序正在运行中'})
    
    data = request.get_json() or {}
    args = data.get('args', [])
    
    thread = threading.Thread(target=runner.run_program, args=(args,))
    thread.daemon = True
    thread.start()
    
    return jsonify({'status': 'success', 'message': '程序已启动'})

@app.route('/api/stop', methods=['POST'])
def stop_program():
    if not program_status['is_running']:
        return jsonify({'status': 'error', 'message': '程序未在运行'})
    
    runner.stop_program()
    return jsonify({'status': 'success', 'message': '程序停止信号已发送'})

@app.route('/api/output')
def get_output():
    return jsonify({'output': program_status['output']})

@app.route('/api/results')
def get_results():
    """获取预测结果文件列表"""
    results = []
    result_dirs = ['presentation/visual_picture', 'presentation/Eda_data']
    
    for dir_path in result_dirs:
        if os.path.exists(dir_path):
            for file in os.listdir(dir_path):
                if file.endswith(('.png', '.jpg', '.csv', '.txt')):
                    results.append({
                        'name': file,
                        'path': os.path.join(dir_path, file),
                        'type': 'image' if file.endswith(('.png', '.jpg')) else 'data'
                    })
    
    return jsonify({'results': results})

@app.route('/api/result/<path:filename>')
def get_result_file(filename):
    """获取具体的结果文件"""
    safe_path = os.path.normpath(filename)
    abs_path = os.path.abspath(safe_path)
    current_dir = os.path.abspath('.')
    
    # 安全检查
    if not abs_path.startswith(current_dir):
        return jsonify({'status': 'error', 'message': '文件路径不安全'}), 403
        
    if os.path.exists(safe_path) and safe_path.startswith('presentation/'):
        return send_file(safe_path)
    else:
        return jsonify({'status': 'error', 'message': '文件不存在'}), 404

# 添加调试信息
@app.route('/api/debug/path')
def debug_path():
    """调试路径信息"""
    info = {
        'current_working_dir': os.getcwd(),
        'script_dir': os.path.dirname(os.path.abspath(__file__)),
        'files_in_cwd': os.listdir('.')
    }
    return jsonify(info)

if __name__ == '__main__':
    # 安装依赖检查
    try:
        import psutil
    except ImportError:
        print("请安装依赖: pip install psutil")
        sys.exit(1)
    
    try:
        import GPUtil
    except ImportError:
        print("GPUtil 未安装，GPU监控将不可用")
    
    # 创建必要的目录
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("🌐 运维平台启动中...")
    print(f"📁 当前工作目录: {os.getcwd()}")
    print(f"📁 脚本所在目录: {os.path.dirname(os.path.abspath(__file__))}")
    print("📱 访问地址: http://127.0.0.1:5000")
    print("🛑 按 Ctrl+C 停止服务器")
    
    app.run(debug=True, host='127.0.0.1', port=5000)