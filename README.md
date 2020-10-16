# py-LabUnAL

``py-LabUnAL_pkg``: LABel UNcertainty using Active Learning

## Installation 

### Windows 

1) [Set up SSH](https://github.com/SenteraLLC/install-instructions/blob/master/ssh_setup.md)
2) Install [conda](https://github.com/SenteraLLC/install-instructions/blob/master/conda.md)
3) Install package

        git clone git@github.com:SenteraLLC/py-LabUnAL.git
        cd py-LabUnAL
        conda env create -f environment.yml
        conda activate py-LabUnAL_conda_venv
        pip install .
   
4) Set up ``pre-commit`` to ensure all commits to adhere to **black** and **PEP8** style conventions.

        pre-commit install
   
#### Linux

1) [Set up SSH](https://github.com/SenteraLLC/install-instructions/blob/master/ssh_setup.md)
2) Install [pyenv](https://github.com/SenteraLLC/install-instructions/blob/master/pyenv.md) and [poetry](https://python-poetry.org/docs/#installation)
3) Install package

        git clone git@github.com:SenteraLLC/py-LabUnAL.git
        cd py-LabUnAL
        pyenv install $(cat .python-version)
        poetry install
        
4) Set up ``pre-commit`` to ensure all commits to adhere to **black** and **PEP8** style conventions.

        poetry run pre-commit install
        
## Usage

Within the correct poetry/conda shell, run ``py-LabUnAL_pkg --help`` to view available CLI commands.
   
## Documentation

This library is documented using Sphinx. To generate documentation, navigate to the *docs/* subfolder,
and run ``make html``.  Make sure you are in the correct conda environment/poetry shell.  Open 
*docs/\_build/html/index.html* with a browser to get more in depth information on the various modules 
and functions within the library.
