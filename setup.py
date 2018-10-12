from setuptools import setup

setup(
    name='torchlib',
    version='0.1',
    packages=['torchlib', 'torchlib.core', 'torchlib.meter', 'torchlib.vision', 'torchlib.vision.model_zoo',
              'torchlib.vision.model_zoo.resnext_features', 'torchlib.vision.vis_tools', 'torchlib.vision.bbox_tools',
              'torchlib.vision.eval_tools', 'torchlib.transforms'],
    url='https://github.com/L1aoXingyu/torchlib',
    license='MIT Licence',
    author='sherlock',
    author_email='sherlockliao01@gmail.com',
    description='lib for training pytorch model'
)
