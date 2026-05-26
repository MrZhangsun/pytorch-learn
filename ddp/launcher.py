import os
import sys
import subprocess


def main():

    world_size = 4

    master_addr = "127.0.0.1"
    master_port = "29500"

    processes = []

    for rank in range(world_size):

        # 复制当前环境变量
        env = os.environ.copy()

        # 设置分布式环境变量
        env["RANK"] = str(rank)
        env["LOCAL_RANK"] = str(rank)
        env["WORLD_SIZE"] = str(world_size)

        env["MASTER_ADDR"] = master_addr
        env["MASTER_PORT"] = master_port

        # 启动 train.py
        process = subprocess.Popen(
            [sys.executable, "dist_demo3.py"],
            env=env
        )

        processes.append(process)

    # 等待所有进程结束
    for process in processes:
        process.wait()

    print("\nAll processes finished")


if __name__ == "__main__":
    main()