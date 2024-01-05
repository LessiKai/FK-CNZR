import setuptools

setuptools.setup(
    name="cnzr",
    author="Someone",
    author_email="some@other.com",
    description="cancer classification",
    version="0.0.1",
    packages=["cnzr"],
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "torch",
        "seaborn",
        "pandas",
        "pathlib",
        "colorama",
        "scikit-learn",
        "hydra-core",
        "tqdm",
        "tensorboard",
        "hydra-submitit-launcher"
    ]
)
