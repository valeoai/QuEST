import os
import urllib.request

os.makedirs('saved_models', exist_ok=True)

model_path = 'http://shape2prog.csail.mit.edu/repo/wrn_40_2_vanilla/ckpt_epoch_240.pth'
model_dir = 'saved_models/wrn_40_2_vanilla'
os.makedirs(model_dir, exist_ok=True)
urllib.request.urlretrieve(model_path, os.path.join(model_dir, model_path.split('/')[-1]))
print(f"Downloaded {model_path.split('repo/')[-1]} to saved_models/")

model_path = 'http://shape2prog.csail.mit.edu/repo/resnet56_vanilla/ckpt_epoch_240.pth'
model_dir = 'saved_models/resnet56_vanilla'
os.makedirs(model_dir, exist_ok=True)
urllib.request.urlretrieve(model_path, os.path.join(model_dir, model_path.split('/')[-1]))
print(f"Downloaded {model_path.split('repo/')[-1]} to saved_models/")

model_path = 'http://shape2prog.csail.mit.edu/repo/resnet110_vanilla/ckpt_epoch_240.pth'
model_dir = 'saved_models/resnet110_vanilla'
os.makedirs(model_dir, exist_ok=True)
urllib.request.urlretrieve(model_path, os.path.join(model_dir, model_path.split('/')[-1]))
print(f"Downloaded {model_path.split('repo/')[-1]} to saved_models/")

model_path = 'http://shape2prog.csail.mit.edu/repo/resnet32x4_vanilla/ckpt_epoch_240.pth'
model_dir = 'saved_models/resnet32x4_vanilla'
os.makedirs(model_dir, exist_ok=True)
urllib.request.urlretrieve(model_path, os.path.join(model_dir, model_path.split('/')[-1]))
print(f"Downloaded {model_path.split('repo/')[-1]} to saved_models/")

model_path = 'http://shape2prog.csail.mit.edu/repo/vgg13_vanilla/ckpt_epoch_240.pth'
model_dir = 'saved_models/vgg13_vanilla'
os.makedirs(model_dir, exist_ok=True)
urllib.request.urlretrieve(model_path, os.path.join(model_dir, model_path.split('/')[-1]))
print(f"Downloaded {model_path.split('repo/')[-1]} to saved_models/")

model_path = 'http://shape2prog.csail.mit.edu/repo/ResNet50_vanilla/ckpt_epoch_240.pth'
model_dir = 'saved_models/ResNet50_vanilla'
os.makedirs(model_dir, exist_ok=True)
urllib.request.urlretrieve(model_path, os.path.join(model_dir, model_path.split('/')[-1]))
print(f"Downloaded {model_path.split('repo/')[-1]} to saved_models/")
