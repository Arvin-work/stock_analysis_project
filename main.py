import sys
import argparse

def main():
    parser = argparse.ArgumentParser(description='股票分析系统')
    parser.add_argument('--simple', action='store_true', help='使用简化模型')
    parser.add_argument('--epochs', type=int, default=80, help='训练轮数')
    parser.add_argument('--data', type=str, default='data/stock_data/hist/600519/20240501_20250905_akshare.csv', 
                       help='数据文件路径')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("📈 股票分析系统启动")
    print("=" * 60)
    print(f"使用模型: {'简化模型' if args.simple else '高级模型'}")
    print(f"训练轮数: {args.epochs}")
    print(f"数据文件: {args.data}")
    print("=" * 60)
    
    # 这里调用你现有的分析代码
    try:
        # 导入并运行你的分析代码
        from analysis.self_pytorch_model import main as analysis_main
        analysis_main()
        
        print("✅ 分析完成！")
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()