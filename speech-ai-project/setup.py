from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements from requirements.txt
def read_requirements():
    requirements = []
    if os.path.exists('requirements.txt'):
        with open('requirements.txt', 'r') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return requirements

setup(
    name='speech-ai-project',
    version='1.0.0',
    author='Speech AI Team',
    author_email='team@speechai.com',
    description='A comprehensive Speech-to-Text and Text-to-Speech AI solution',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/speech-ai-project',
    
    # Package configuration
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    
    # Include additional files
    package_data={
        '': ['*.yaml', '*.yml', '*.txt', '*.md'],
        'config': ['*.yaml', '*.yml'],
        'data': ['*'],
    },
    include_package_data=True,
    
    # Dependencies
    install_requires=read_requirements(),
    
    # Optional dependencies
    extras_require={
        'dev': [
            'pytest>=6.2.0',
            'pytest-cov>=3.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'isort>=5.10.0',
        ],
        'notebook': [
            'jupyter>=1.0.0',
            'matplotlib>=3.5.0',
            'seaborn>=0.11.0',
        ],
        'audio-quality': [
            'resampy>=0.2.2',
            'noisereduce>=2.0.0',
        ],
        'gpu': [
            'torch>=1.11.0+cu113',
            'torchaudio>=0.11.0+cu113',
        ]
    },
    
    # Entry points for command-line tools
    entry_points={
        'console_scripts': [
            'speech-ai=main:main',
            'stt=main:main',
            'tts=main:main',
        ],
    },
    
    # Metadata
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Multimedia :: Sound/Audio :: Speech',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    
    # Python version requirements
    python_requires='>=3.8',
    
    # Keywords for package discovery
    keywords='speech recognition, text to speech, AI, machine learning, audio processing',
    
    # Project URLs
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/speech-ai-project/issues',
        'Source': 'https://github.com/yourusername/speech-ai-project',
        'Documentation': 'https://speech-ai-project.readthedocs.io/',
    },
    
    # Additional options
    zip_safe=False,  # Don't zip the package
    platforms=['any'],
    
    # Test suite
    test_suite='tests',
    tests_require=[
        'pytest>=6.2.0',
        'pytest-cov>=3.0.0',
    ],
)