import h5py
import os
from omnigibson.learning.utils.obs_utils import rgbd_vid_to_pcd

# 配置
data_path = "/raid/ljh/BEHAVIOR-1K/datasets"
task_id = 0  

# 每个 Task 里的 demo 编号是从 10, 20, 30 ... 到 3000
local_demo_indices = range(10, 3001, 10)

for local_idx in local_demo_indices:
    # 规则：Task ID * 10000 + 局部编号
    # 例如：Task 3, Index 10 -> 30010 -> 文件名 episode_00030010.parquet
    demo_id = task_id * 10000 + local_idx

    # 1. 构造输入文件路径以检查是否存在
    input_parquet_path = os.path.join(
        data_path, 
        "2025-challenge-demos", 
        "data", 
        f"task-{task_id:04d}", 
        f"episode_{demo_id:08d}.parquet" # 这里会自动变成 00030010
    )

    # 2. 检查文件是否存在
    if not os.path.exists(input_parquet_path):
        print(f"[Skip] {input_parquet_path} 不存在，跳过。") 
        continue

    print(f"--------------------------------------------------")
    print(f"[Running] 正在处理 Task {task_id} - Demo {demo_id} (Local Index: {local_idx})")

    output_hdf5_path = os.path.join(
        data_path, 
        "pcd_vid", 
        f"task-{task_id:04d}", 
        f"episode_{demo_id:08d}.hdf5"
    )

    # 检查输出文件是否存在且有效
    if os.path.exists(output_hdf5_path):
        try:
            # 尝试打开文件检查是否完好
            with h5py.File(output_hdf5_path, "r") as f:
                # 检查关键数据是否存在
                if "data/demo_0/robot_r1::fused_pcd" in f:
                    print(f"[Skip] {output_hdf5_path} 已存在且有效，跳过。")
                    continue
                else:
                    print(f"[Warn] {output_hdf5_path} 存在但在缺少关键key，将重新生成。")
        except OSError:
            print(f"[Warn] {output_hdf5_path} 存在但已损坏，将重新生成。")

    try:
        # 3. 执行转换
        rgbd_vid_to_pcd(
            data_folder=data_path,
            task_id=task_id,
            demo_id=demo_id,  
            episode_id=0,     # 输出 HDF5 内部的组名，通常保持 0 即可
            pcd_range=(-0.2, 1.5, -1.5, 1.5, 0.2, 1.5),
            batch_size=1000,
            use_fps=True,
        )
        print(f"[Success] Demo {demo_id} 处理成功。")

        f_pcd_vid = h5py.File(output_hdf5_path, "r", libver="latest", swmr=True)
        pcd_vid = f_pcd_vid["data/demo_0/robot_r1::fused_pcd"][:]
        print(pcd_vid.shape)

    except Exception as e:
        print(f"[Error] 处理 Demo {demo_id} 时发生错误: {e}")

print("--------------------------------------------------")
print("所有任务处理完毕。")