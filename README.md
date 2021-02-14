# deLIBErate
deLIBErate is a repository of code to assist in the use and analysis of the Lithium-Ion Battery Electrolyte (LIBE) dataset.

The repository is primarily comprised of Jupyter Notebooks, with some utility functions included.

## Requirements

The notebooks included here require the following libraries. We recommend that you use [conda](https://docs.conda.io/en/latest/) to install, with Python version >= 3.6.

- Numpy
- SciPy
- Matplotlib
- Seaborn
- NetworkX
- OpenBabel / pybel
- [monty](https://github.com/materialsvirtuallab/monty)
- [pymatgen](https://github.com/materialsproject/pymatgen)

While not used directly in this repository, the following libraries are necessary to completely replicate the workflow used to generate LIBE:

- [custodian](https://github.com/materialsproject/custodian)
- [atomate](https://github.com/hackingmaterials/atomate)
- [bondnet](https://github.com/mjwen/bondnet)

## Installation

In order to use the utility code provided in `src`, run `python setup.py develop`. After running this command, all Jupyter notebooks should run all code without error.