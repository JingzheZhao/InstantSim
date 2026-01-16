import os

# ---------------- 配置区域 ----------------
# 这里填写你要修改的文件夹路径 (注意斜杠的方向，Windows用反斜杠可能需要转义，或者直接用 /)
# 假设你的结构是 dataset/train_A/rectangle
target_folder = "E:/InstantSim/dataset/test_B/voronoi"

# 起始偏移量 (你想让第1张图变成第几号？这里填 901)
start_number = 2000


# ----------------------------------------

def batch_rename():
    # 1. 获取所有文件
    if not os.path.exists(target_folder):
        print(f"❌ 错误：找不到文件夹 {target_folder}")
        return

    files = [f for f in os.listdir(target_folder) if f.endswith(('.png', '.jpg'))]

    # 2. 极其重要：先排序！确保 0001 对应 901，0002 对应 902
    files.sort()

    print(f"📂 正在处理 {target_folder}，共 {len(files)} 个文件...")

    # 3. 循环重命名
    count = 0
    for i, filename in enumerate(files):
        # 构造旧路径
        old_path = os.path.join(target_folder, filename)

        # 构造新名字：当前索引 + 起始号 (例如：0 + 901 = 901)
        new_num = start_number + i

        # 格式化为 4位数字，例如 0901.png
        new_filename = f"{new_num:04d}.png"
        new_path = os.path.join(target_folder, new_filename)

        # 执行重命名
        try:
            os.rename(old_path, new_path)
            # 打印前几个看看效果，避免刷屏
            if i < 5:
                print(f"✅ Renamed: {filename} -> {new_filename}")
            count += 1
        except Exception as e:
            print(f"❌ 重命名 {filename} 失败: {e}")

    print(f"🎉 完成！共重命名了 {count} 个文件。")
    print(f"最后一张图应该是: {start_number + count - 1:04d}.png")


if __name__ == "__main__":
    # 为了防止意外，建议先备份一下 rectangle 文件夹，然后运行
    batch_rename()