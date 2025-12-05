import os
import time
from huggingface_hub import snapshot_download

# 1. 确保设置了镜像（如果你依然需要用镜像）
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 2. 设置路径
repo_id = "behavior-1k/2025-challenge-demos"
local_dir = "/raid/ljh/BEHAVIOR-1K/datasets/2025-challenge-demos"

print(f"目标仓库: {repo_id}")
print(f"本地路径: {local_dir}")

# 3. 循环重试机制
while True:
    try:
        print("\n[开始] 尝试发起下载...")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
            # max_workers=2 是关键！默认是 8，太高会导致不稳定或被 429 封锁
            max_workers=2,
            # 忽略一些不重要的匹配模式，也许能快一点（可选）
            # ignore_patterns=["*.gitattributes", "README.md"], 
        )
        print("\n[成功] 所有文件下载完毕！")
        break  # 下载成功，跳出循环

    except Exception as e:
        print(f"\n[失败] 发生错误: {e}")
        # print("[等待] 网络不稳定，休息 10 秒后自动重试...")
        # time.sleep(1)