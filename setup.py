
from setuptools import setup

from dsp4bats import __version__

setup(name='dsp4bats',
    version=__version__,
    description='Digital Sound Processing for bats, a part of the CloudedBats.org project.',
    url='https://github.com/cloudedbats/cloudedbats_dsp',
    author='Arnold Andreasson',
    author_email='info@cloudedbats.org',
    license='MIT',
    packages=['dsp4bats'],
    install_requires=[
        'numpy', 
        'matplotlib', 
        'pandas', 
        'librosa', 
    ],
    zip_safe=False)