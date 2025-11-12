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
        disk = psutil.disk_usage('/')
        
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

runner = ProgramRunner()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    return jsonify(program_status)

@app.route('/api/system')
def get_system_status():
    """获取系统状态"""
    system_info = get_system_info()
    return jsonify(system_info)

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
    if os.path.exists(safe_path) and safe_path.startswith('presentation/'):
        return send_file(safe_path)
    else:
        return jsonify({'status': 'error', 'message': '文件不存在'}), 404

if __name__ == '__main__':
    # 安装依赖检查
    try:
        import psutil
        import GPUtil
    except ImportError:
        print("请安装依赖: pip install psutil GPUtil")
        sys.exit(1)
    
    # 创建必要的目录
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    print("🌐 股票分析系统控制面板启动中...")
    print("📱 访问地址: http://127.0.0.1:5000")
    print("🛑 按 Ctrl+C 停止服务器")
    
    app.run(debug=True, host='127.0.0.1', port=5000)