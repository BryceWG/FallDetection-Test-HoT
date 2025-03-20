import os
import numpy as np
import json
import pandas as pd
import argparse

def ensure_dir(directory):
    """
    确保目录存在，如果不存在则创建
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_2d_npz(input_file, output_file, start_frame, end_frame):
    """
    处理2D姿态的npz文件
    """
    try:
        data = np.load(input_file)
        # 检查可用的键名
        print(f"2D NPZ文件 {input_file} 包含以下键：{list(data.keys())}")
        
        # 从reconstruction键中获取数据
        if 'reconstruction' in data:
            keypoints = data['reconstruction'][0][start_frame:end_frame+1]  # 注意这里的[0]，因为数据格式是(1, frames, joints, 2)
            np.savez_compressed(output_file, reconstruction=np.array([keypoints]))  # 保持相同的数据结构
            print(f"成功保存2D姿态数据，形状为: {keypoints.shape}")
        else:
            raise KeyError(f"未找到'reconstruction'键，可用的键为: {list(data.keys())}")
            
    except Exception as e:
        print(f"处理2D NPZ文件时出错: {str(e)}")
        raise

def process_2d_json(input_file, output_file, start_frame, end_frame):
    """
    处理2D姿态的json文件
    """
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # 提取指定帧范围的数据
        filtered_data = []
        
        # 由于原始数据中idx总是0，我们使用enumerate来获取实际的帧索引
        for frame_idx in range(start_frame, end_frame + 1):
            if frame_idx < len(data):
                frame_data = data[frame_idx]
                filtered_data.append({
                    'idx': frame_idx - start_frame,  # 重新编号，从0开始
                    'keypoints': frame_data['keypoints']
                })
        
        if not filtered_data:
            print(f"警告：在指定范围 ({start_frame}-{end_frame}) 内未找到有效帧")
        else:
            print(f"处理帧范围: {start_frame} - {end_frame}, 总帧数: {len(filtered_data)}")
        
        with open(output_file, 'w') as f:
            json.dump(filtered_data, f, indent=4)
        print(f"成功保存JSON数据，包含 {len(filtered_data)} 帧")
    except Exception as e:
        print(f"处理JSON文件时出错: {str(e)}")
        raise

def process_3d_npz(input_file, output_file, start_frame, end_frame):
    """
    处理3D姿态的npz文件
    """
    try:
        data = np.load(input_file)
        # 检查可用的键名
        print(f"3D NPZ文件 {input_file} 包含以下键：{list(data.keys())}")
        
        # 从reconstruction键中获取数据
        if 'reconstruction' in data:
            keypoints = data['reconstruction'][start_frame:end_frame+1]
            np.savez_compressed(output_file, reconstruction=keypoints)
            print(f"成功保存3D姿态数据，形状为: {keypoints.shape}")
        else:
            raise KeyError(f"未找到'reconstruction'键，可用的键为: {list(data.keys())}")
            
    except Exception as e:
        print(f"处理3D NPZ文件时出错: {str(e)}")
        raise

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='处理姿态数据分割工具')
    parser.add_argument('--csv', type=str, required=True, help='CSV文件路径')
    parser.add_argument('--base-dir', type=str, default='./demo/output', help='输入数据根目录路径')
    args = parser.parse_args()
    
    try:
        # 读取CSV文件
        print(f"正在读取CSV文件: {args.csv}")
        df = pd.read_csv(args.csv)
        
        # 只处理has_fall为1的记录
        fall_segments = df[df['has_fall'] == 1]
        print(f"找到 {len(fall_segments)} 个跌倒片段需要处理")
        
        for idx, row in fall_segments.iterrows():
            try:
                video_id = row['video_id']
                start_frame = row['fall_start_frame']
                end_frame = row['fall_end_frame']
                
                print(f"\n处理视频 {video_id} ({idx + 1}/{len(fall_segments)})")
                print(f"帧范围: {start_frame} - {end_frame}")
                
                # 构建文件路径
                video_dir = os.path.join(args.base_dir, video_id)
                
                # 创建splits目录
                input_2d_splits_dir = os.path.join(video_dir, 'input_2D', 'splits')
                output_3d_splits_dir = os.path.join(video_dir, 'output_3D', 'splits')
                ensure_dir(input_2d_splits_dir)
                ensure_dir(output_3d_splits_dir)
                
                # 处理2D NPZ文件
                input_2d_npz = os.path.join(video_dir, 'input_2D', 'input_keypoints_2d.npz')
                output_2d_npz = os.path.join(input_2d_splits_dir, 'fall_keypoints_2d.npz')
                if os.path.exists(input_2d_npz):
                    process_2d_npz(input_2d_npz, output_2d_npz, start_frame, end_frame)
                else:
                    print(f"警告：2D NPZ文件不存在: {input_2d_npz}")
                
                # 处理2D JSON文件
                input_2d_json = os.path.join(video_dir, 'input_2D', 'keypoints_2d.json')
                output_2d_json = os.path.join(input_2d_splits_dir, 'fall_keypoints_2d.json')
                if os.path.exists(input_2d_json):
                    process_2d_json(input_2d_json, output_2d_json, start_frame, end_frame)
                else:
                    print(f"警告：2D JSON文件不存在: {input_2d_json}")
                
                # 处理3D NPZ文件
                input_3d_npz = os.path.join(video_dir, 'output_3D', 'output_keypoints_3d.npz')
                output_3d_npz = os.path.join(output_3d_splits_dir, 'fall_keypoints_3d.npz')
                if os.path.exists(input_3d_npz):
                    process_3d_npz(input_3d_npz, output_3d_npz, start_frame, end_frame)
                else:
                    print(f"警告：3D NPZ文件不存在: {input_3d_npz}")
                
            except Exception as e:
                print(f"处理视频 {video_id} 时出错: {str(e)}")
                continue
            
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()
