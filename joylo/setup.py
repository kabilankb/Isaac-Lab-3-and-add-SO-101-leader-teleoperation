from setuptools import setup, find_packages

setup(
    name="joylo",
    version="0.1.0",
    description="JoyLo teleoperation for YAM bimanual robot control",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    license="MIT",
    install_requires=[
        "numpy",
        "pyyaml", 
        "dynamixel-sdk",
        "joycon-python",
        "pybullet",
        "portal",
        "tyro",
        "pyglm",
        "hid",
        "typing_extensions==4.12.2",
    ],
)
