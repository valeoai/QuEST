# QuEST: Quantized embedding space for transferring knowledge
This repository contains the code for [QuEST](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123660171.pdf), an approach for knowledge distillation.

[![Video](http://img.youtube.com/vi/bJyJkAhjp88/0.jpg)](https://www.youtube.com/watch?v=bJyJkAhjp88)

#### Config files
Config files named as <*TeacherNetwork*> or <*TeacherNetwork_StudentNetwork*>.

To evaluate the teacher or to learn K-means, use <*TeacherNetwork*> eg. WRN40w2.py, RN56.py, VGG13.py.

To train or evaluate student model, use <*TeacherNetwork_StudentNetwork*>, eg. WRN40w2_WRN16w2.py in config folder.
#### Evaluate teacher network:
```
python scripts/main_classification.py with config=Config_CIFAR100.<TeacherNetwork> evaluate=True
# output logs are stored at ./experiments/Config_CIFAR100/<TeacherNetwork>/
```
#### Learn K-means codebook:
```
python scripts/main_classification.py with config=Config_CIFAR100.<TeacherNetwork> kmeans=4096
# The learned codebook will be stored at ./experiments/VQ/Config_CIFAR100/<TeacherNetwork>/
```
#### Train student:
```
python scripts/main_classification.py with config=Config_CIFAR100.<TeacherNetwork_StudentNetwork>
# The trained model files and logs are at ./experiments/KD/Config_CIFAR100/<TeacherNetwork_StudentNetwork>/
```

#### Evaluate student network:
```
python scripts/main_classification.py with config=Config_CIFAR100.<TeacherNetwork_StudentNetwork> evaluate=True
# It uses the last saved model from ./experiments/Config_CIFAR100/<TeacherNetwork_StudentNetwork>/.
# To evaluate the best model give 'best=True' as command line argument.
# To evaluate any model, create a config file with the model paths and other info. For example see <TeacherNetwork> config files.
```
