# Python Libraries for Incomplete Discrete Choice (IDC) Models

This repository collects Python libraries to estimate discrete choice models. Currently, it contains the following files.

- `idclib.py`: The main library
- `examples.py`: Additional files for the examples used in the Jupyter notebooks.

# Modules used

Depending on your environment, you may need to install the following modules.
- CVXPY (https://www.cvxpy.org/install/): A module for convex optimization. Installation: `pip install cvxpy`
- down (https://pypi.org/project/gdown/): A module for downloading files from Google Drive. Installation: `pip install gdown`

# Tutorials

The following Jupyter Notebooks (developed using Google Colab) review the basics of discrete choice models and illustrate how to use the library.
- `CompleteModel.ipynb`: A review of a binary choice model;
- `IncompleteModel.ipynb`: A review of a two-player discrete game;
- `ModelPrediction.ipynb`: Explains how to represent a DC model as a graph;
- `Inequalities.ipynb`: Explains how to obtain the sharp identifying restrictions using the `idc` library;
- `HypothesisTests.ipynb`: Explains how to conduct hypothesis tests using universal inference methods.
- Additional examples
  - `Ex_Panel.ipynb`: Panel binary choice model
  - `Ex_OrderedChoice.ipynb`: Ordered choice with a set-valued control function
  - `Ex_HeterogeneousChoiceSets.ipynb`: Multinomial choice with heterogeneous choice sets
