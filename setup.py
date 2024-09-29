from setuptools import setup, find_packages

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    "charset-normalizer",
    "matplotlib",
    "openai",
    'torch<=2.0.0',
    'numpy',
    'ray>=1.1.0',
    'tensorboard>=1.14.0',
    'tensorboardX>=1.6',
    'setproctitle',
    'psutil',
    'pyyaml',
    "gym==0.23.1",
    "omegaconf",
    "termcolor",
    "hydra-core>=1.1",
    "pyvirtualdisplay",
]

# Installation operation
setup(
    name="icpl",
    author="Jason Ma",
    version="1.0",
    description="Icpl",
    keywords=["llm", "rl"],
    include_package_data=True,
    python_requires=">=3.7",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages("."),
)

