from setuptools import setup

description = "Bounding box augmentation for PyTorch with Google's Brain Team augmentation policies"

with open('README.md', encoding='utf-8') as f:
    long_description = '\n' + f.read()

setup(
    name='bbaug',
    version='0.2.0',
    author='Harpal Sahota',
    author_email='harpal28sahota@gmail.com',
    description=description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/harpalsahota/bbaug',
    packages=[
        'bbaug',
        'bbaug.augmentations',
        'bbaug.policies',
        'bbaug.visuals',
    ],
    python_requires='>=3.5',
    install_requires=[
        'imgaug',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ]
)