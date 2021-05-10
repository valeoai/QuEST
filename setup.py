from setuptools import setup
from setuptools import find_packages

setup(
    name='QuEST',
    description='QUEST: Quantized embedding space for transferring knowledge',
    packages=find_packages(),
    install_requires=["tqdm",
                      "sacred",
                      "numpy",
                      "torch",
                      "torchvision",
                      "Pillow"]
    )
