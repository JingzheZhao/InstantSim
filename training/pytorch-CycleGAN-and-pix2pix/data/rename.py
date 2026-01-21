import os


target_folder = "E:/InstantSim/dataset/test_B/voronoi"

start_number = 2000



def batch_rename():
    if not os.path.exists(target_folder):
        print(f"error: can't find folder {target_folder}")
        return

    files = [f for f in os.listdir(target_folder) if f.endswith(('.png', '.jpg'))]

    files.sort()

    print(f" processing {target_folder}，total {len(files)} ")

    count = 0
    for i, filename in enumerate(files):
        old_path = os.path.join(target_folder, filename)

        new_num = start_number + i

        new_filename = f"{new_num:04d}.png"
        new_path = os.path.join(target_folder, new_filename)

        try:
            os.rename(old_path, new_path)
            if i < 5:
                print(f"Renamed: {filename} -> {new_filename}")
            count += 1
        except Exception as e:
            print(f" rename {filename} failed: {e}")

    print(f"🎉 done！total {count}")
    print(f"last image is: {start_number + count - 1:04d}.png")


if __name__ == "__main__":
    batch_rename()