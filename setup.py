import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='pykitml',
    version='0.0.1',
    author='RainingComputers',
    author_email='vishnu.vish.shankar@gmail.com',
    description='Machine Learning library written in Python and NumPy.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    #url='https://github.com/pypa/sampleproject',
    packages=setuptools.find_packages(exclude=['docs', 'tests']),
    python_requires='>=3.5',
    install_requires=[
        'numpy', 'matplotlib', 'tqdm', 'graphviz'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Development Status :: 1 - Planning',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    keywords='pykitml'
)