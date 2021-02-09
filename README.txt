Config files named as <TeacherNetwork> or <TeacherNetwork_StudentNetwork>.
<TeacherNetwork>: To evaluate the teacher or to learn K-means. Eg. WRN40w2.py, RN56.py, VGG13.py in config folder.
<TeacherNetwork_StudentNetwork>: To train or evaluate student model. Eg. WRN40w2_WRN16w2.py in config folder.

Evaluate teacher network:
python scripts/main_classification.py with config=Config_CIFAR100.<TeacherNetwork> evaluate=True
# output logs are stored at ./experiments/Config_CIFAR100/<TeacherNetwork>/
Learn K-means codebook:
python scripts/main_classification.py with config=Config_CIFAR100.<TeacherNetwork> kmeans=4096
# The learned codebook will be stored at ./experiments/VQ/Config_CIFAR100/<TeacherNetwork>/

Train student:
python scripts/main_classification.py with config=Config_CIFAR100.<TeacherNetwork_StudentNetwork>
# The trained model files and logs are at ./experiments/KD/Config_CIFAR100/<TeacherNetwork_StudentNetwork>/

Evaluate student network:
python scripts/main_classification.py with config=Config_CIFAR100.<TeacherNetwork_StudentNetwork> evaluate=True
# It uses the last saved model from ./experiments/Config_CIFAR100/<TeacherNetwork_StudentNetwork>/.
# To evaluate the best model put 'best=True' with the above command.

To evaluate any model by path create a config file with model paths and other info see eval_WRN for example. Then run it as,
python scripts/main_classification.py with config=Config_CIFAR100.eval_<*> evaluate=True
# eval_<*> should specify the 'feature_extractor' and 'classifier' networks' paths.
