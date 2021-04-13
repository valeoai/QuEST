from __future__ import print_function

import shutil
from pathlib import Path



def copy_images_for_scr_to_dst(filename, src_dir, dst_dir):
    with open(filename) as f:
        contents = f.readlines()

    num_files = len(contents)
    for i, file in enumerate(contents):
        file = file.strip()

        src_file = Path(src_dir) / file
        assert src_file.is_file()

        dst_file = Path(dst_dir) / file
        dst_file_dir = dst_file.parents[0]
        dst_file_dir.mkdir(parents=True, exist_ok=True)

        print(f'==> [{i} / {num_files}] {src_file} ==> {dst_file}')
        shutil.copy(str(src_file), str(dst_file))
        assert dst_file.is_file()


if __name__ == '__main__':
    src_dir = './datasets/MITscenes/Images'
    dst_dir = './datasets/MITscenes/'

    train_filename = './datasets/MITscenes/TrainImages.txt'
    copy_images_for_scr_to_dst(train_filename, src_dir, dst_dir + 'train')

    test_filename = './datasets/MITscenes/TestImages.txt'
    copy_images_for_scr_to_dst(test_filename, src_dir, dst_dir + 'test')
