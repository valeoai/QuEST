set -e
sudo docker build -t test_image_spyros .
sudo docker run --shm-size=2048M --gpus "device=1" -v /media/data/datasets:/datasets_local -v test_image_spyros bash -c "python scripts/main_classification.py with config=Config_CIFAR100.WRN40w2 evaluate=True data_dir=./datasets/CIFAR/"
