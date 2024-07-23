# GML_for_ALSA

Gradual Machine Learning for Aspect-level Sentiment Analysis

## Table of Contents
- [GML\_for\_ALSA](#gml_for_alsa)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Running Jupyter Notebook](#running-jupyter-notebook)
  - [Project Structure](#project-structure)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

## Overview
Gradual Machine Learning for Aspect-level Sentiment Analysis (GML for ALSA) is a project aimed at providing advanced techniques for sentiment analysis at the aspect level. This project utilizes gradual machine learning methods to improve the accuracy and efficiency of sentiment analysis tasks.

## Installation
To install the project, clone the repository and set up the environment using the following commands:

```bash
git clone https://github.com/benchaddha/GML_for_ALSA.git
cd GML_for_ALSA
python -m venv myenv
source myenv/bin/activate   # On Windows use `myenv\Scripts\activate`
pip install -r requirements.txt
```

## Usage
To run the GML for ALSA, execute the main script. Ensure you have the necessary data files in the `data` directory.

```bash
python gml.py
```
Outputs are verbose so run the following to update output.txt with the terminal output in real time: 
```bash
python gml.py &> output.txt
```

### Running Jupyter Notebook
To explore the API documentation and examples, run the Jupyter Notebook:

```bash
jupyter notebook "API Doc.ipynb"
```

## Project Structure
- `approximate_probability_estimation.py`: Contains the `ApproximateProbabilityEstimation` class for calculating approximate probabilities.
- `data_structures.py`: Defines the data structures used for variables and features.
- `easy_instance_labeling.py`: Implements the `EasyInstanceLabeling` class for marking easy instances.
- `evidence_select.py`: Provides methods for selecting evidence.
- `evidential_support.py`: Contains classes for calculating evidential support.
- `get_data.py`: Script for loading and processing data.
- `gml.py`: Main script for running the GML process.
- `gml_utils.py`: Utility functions for the GML process.
- `helper.py`: Helper functions and classes.
- `numbskull/`: Directory containing the Numbskull library for factor graph inference and learning.
- `data/`: Directory containing data files required for the project.


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Stephen Bach for the initial implementation of the Numbskull library.
- Chenyu Wang for initial GML implementation
- The paper *Gradual Machine Learning for Aspect-level Sentiment Analysis* by Yanyan Wang, Qun Chen, Jiquan Shen, Boyi Hou, Murtadha H. M. Ahmed, Zhanhuai Li
