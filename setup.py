import setuptools

with open("yaep/README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="yaep",
    version="0.0.1",
    author="Zylo117",
    author_email="",
    description="Pytorch EfiicientDet implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Skarsh/Yet-Another-EfficientDet-Pytorch",
    packages=setuptools.find_packages(),
    install_requires=[
        "scikit-build",
        "cmake",
        "pycocotools",
        "numpy",
        "opencv-python",
        "tqdm",
        "tensorboard",
        "tensorboardX",
        "pyyaml",
        "webcolors",
        "torch==1.4.0",
        "torchvision==0.5.0"
    ]
)
