from setuptools import setup

readme = open('README.rst').read()

setup(name='dtoolai',
      version='0.1.1',
      description='Reproducible Deep Learning tools and examples',
      long_description=readme,
      long_description_content_type='text/x-rst',
      url='http://github.com/JIC-CSB/dtoolai',
      author='Matthew Hartley',
      author_email='Matthew.Hartley@jic.ac.uk',
      license='MIT',
      install_requires=[
        'dtoolcore',
        'dtool-http',
        'click',
        'pillow',
        'torch',
        'torchvision',
      ],
      packages=['dtoolai'],
      entry_points={
        'console_scripts': [
          'dtoolai-provenance = dtoolai.utils:print_provenance',
          'create-image-dataset-from-dirtree = dtoolai.utils:image_dataset_from_dirtree_cli'
        ]
      },
      zip_safe=False)
