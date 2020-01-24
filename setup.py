from setuptools import setup

setup(name='dtoolai',
      version='0.1',
      description='dtool AI utils',
      url='http://github.com',
      author='Matthew Hartley',
      author_email='Matthew.Hartley@jic.ac.uk',
      license='MIT',
      install_requires=[
        'dtoolcore',
        'click',
        'pillow',
        'torch',
        'torchvision',
      ],
      packages=['dtoolai'],
      zip_safe=False)
