import os
import random
from glob import glob
import shutil 

file_list = glob(r"118.안면 인식 에이징(aging) 이미지 데이터\01-1.정식개방데이터\Validation\01.원천데이터\*/*.png")

for file in file_list:
    age = int(os.path.basename(file).split("_")[2])
    if age < 20:
        dst = os.path.join("118.안면 인식 에이징(aging) 이미지 데이터", "Total", "Child", os.path.basename(file))
        os.rename(file, dst)
    else:
        dst = os.path.join("118.안면 인식 에이징(aging) 이미지 데이터", "Total", "Adult", os.path.basename(file))
        os.rename(file, dst)
        
        
base_input = r"118.안면 인식 에이징(aging) 이미지 데이터\Total"
base_output = "Datasets"
splits = ["Train", "Val", "Test"]
split_ratios = [0.7, 0.2, 0.1]

for category in ["Adult", "Child"]:
    files = glob(os.path.join(base_input, category, "*.png"))
    random.shuffle(files)

    total = len(files)
    train_end = int(total * split_ratios[0])
    val_end = train_end + int(total * split_ratios[1])

    split_files = {
        "Train": files[:train_end],
        "Val": files[train_end:val_end],
        "Test": files[val_end:]
    }

    for split in splits:
        output_dir = os.path.join(base_output, split, category)
        os.makedirs(output_dir, exist_ok=True)

        for file in split_files[split]:
            dst = os.path.join(output_dir, os.path.basename(file))
            shutil.move(file, dst)  
