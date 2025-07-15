from setuptools import setup, find_packages

with open("requirement.txt") as f:
    requirements = f.read().splitlines()

setup(
    name='EngToBengaliTranslation',
    version='1.0.0',
    description='English to Bengali Translation Model',
    author='Md Asifur Rahman',
    packages=find_packages(),
    install_requires=requirements,
    python_requires='>=3.6',
    entry_points={
        'console_scripts': [
            'train-model = EngToBengaliTranslation.train:main',
            'test-model = EngToBengaliTranslation.test:main',
        ],
    },
)