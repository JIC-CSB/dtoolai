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
      entry_points={
        'console_scripts': [
          'dtoolai-provenance = dtoolai.utils:print_provenance',
          'create-image-dataset-from-dirtree = dtoolai.utils:image_dataset_from_dirtree'
        ]
      },
      zip_safe=False)
