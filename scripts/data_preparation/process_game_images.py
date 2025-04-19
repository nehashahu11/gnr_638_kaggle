import os
import shutil
import random
from tqdm import tqdm

def split_dataset(hr_dir, lr_dir, output_dir, val_ratio=0.2, seed=42):
    random.seed(seed)

    
    for split in ['train', 'val']:
        for subfolder in ['hr', 'lr']:
            os.makedirs(os.path.join(output_dir, split, subfolder), exist_ok=True)

    
    hr_files = sorted(os.listdir(hr_dir))
    lr_files = sorted(os.listdir(lr_dir))

    assert len(hr_files) == len(lr_files), "HR and LR folder should contain same number of files"
    
    files = list(zip(hr_files, lr_files))
    random.shuffle(files)

    val_size = int(len(files) * val_ratio)
    val_files = files[:val_size]
    train_files = files[val_size:]

    for split_name, split_files in [('train', train_files), ('val', val_files)]:
        for hr_file, lr_file in tqdm(split_files, desc=f"Copying to {split_name}"):
            shutil.copy(os.path.join(hr_dir, hr_file), os.path.join(output_dir, split_name, 'hr', hr_file))
            shutil.copy(os.path.join(lr_dir, lr_file), os.path.join(output_dir, split_name, 'lr', lr_file))

    


split_dataset(
    hr_dir='train-kaggle/hr',
    lr_dir='train-kaggle/lr',
    output_dir='output-split',
    val_ratio=0.2  
)
