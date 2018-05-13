from setuptools import setup, find_packages

print(find_packages())
setup(
    name='slide_analysis_nn',
    version='1',
    packages=find_packages(),
    install_requires=['matplotlib',
                      'werkzeug',
                      'keras',
                      'tensorflow',
                      'opencv-python',
                      'falcon',
                      'numpy',
                      'h5py',
                      'openslide-python',
                      'shapely',
                      'requests',
                    ],
    url='',
    license='MIT',
    author='vozman',
    author_email='vozman@yandex.ru',
    description='',
    include_package_data=True,
)
