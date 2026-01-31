from setuptools import setup, find_packages

setup(
    name='diffusion-policy',
    packages=find_packages(),
    version='1.0.0',
    license='MIT',
    description='Diffusion Policy - PyTorch Lightning implementation for robot learning',
    author='DD-PPO Team',
    long_description_content_type='text/markdown',
    keywords=[
        'artificial intelligence',
        'generative models',
        'diffusion models',
        'robot learning',
        'policy learning',
    ],
    install_requires=[
        'einops',
        'numpy',
        'scipy',
        'torch>=2.0',
        'pytorch-lightning>=2.0',
        'tqdm',
    ],
    extras_require={
        'visualization': [
            'open3d',
            'matplotlib',
        ],
        'logging': [
            'wandb',
            'tensorboard',
        ],
        'dev': [
            'pytest',
            'pytest-cov',
        ],
    },
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    entry_points={
        'console_scripts': [
            'train-diffusion-policy=scripts.train:main',
            'evaluate-diffusion-policy=scripts.evaluate:main',
        ],
    },
)
