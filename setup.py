from setuptools import setup, find_packages

setup(
    name='yamlab',
    version='0.1.0',
    description='YAMLab: A Bimanual YAM Robot Simulation Framework for Data Collection, Data Generation, and Large-Scale Parallel Evaluation',
    author='Tianyuan Dai',
    author_email='tydai@utexas.edu',
    url='https://github.com/ARISE-Initiative/yamlab',
    license='MIT',
    packages=find_packages(exclude=["joylo*"]),
    include_package_data=True,
    package_data={
        'yamlab': [
            'robot/**/*.usd',
            'robot/**/*.json',
            'robot/**/*.yaml',
            'robot/**/*.stl',
            'robot/**/.asset_hash',
            'configs/*.yaml',
        ],
    },
)
