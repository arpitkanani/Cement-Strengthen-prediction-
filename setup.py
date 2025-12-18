from setuptools import setup, find_packages
from typing import List
import warnings
warnings.filterwarnings("ignore")

HYPHON_E_DOT = '-e .'

def get_requirements(file_path:str)->List[str]:
    requirements = []
    with open(file_path) as f:
      requirements=f.readlines()
      requirements = [req.replace("\n", "") for req in requirements]
      if HYPHON_E_DOT in requirements:
            requirements.remove(HYPHON_E_DOT)
    return requirements
        


setup(
    name='Cement Strenth prediction',
    version='0.1',
    author='Arpit Kanani',
    author_email="kananiarpit1411@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)