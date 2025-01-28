# HA-ES: Hardware-Aware Ensemble Selection
One of the concerns with currently existing ensemble selection algorithms is the size of the ensemble, which can affect the inference speed of the trained model. To address this issue, we introduce our novel approach: hardware-aware ensemble selection (HA-ES), which focuses on finding a balance in the performance and complexity trade-off inherent in ensembling.

To evaluate our approach and compare it to the existing algorithms, we use TabRepo, which provides prediction probabilities for over 100 ML problems. We use this data to efficiently evaluate and compare the ensemble selection techniques.


## Set-Up
This set-up guide expects a Linux system. Further the code is only tested with Python version 3.10.14.

### (Optional) Create venv
It's good practice to use a virtual environment. This isolates your project dependencies from global Python installations. This is how you create a virtual environment in your project directory:
- `python3 -m venv venv`
To activate it use:
- `source venv/bin/activate`


### Dependencies
From the project root run the following commands to install the dependencies (`-e` to automatically install changes made to the code of the dependencies)
- `pip install -r requirements.txt`
- `python3 -m pip install -e extern/tabrepo`
- `python3 -m pip install -e extern/phem`

### Run test
To run the experiments use
- `python3 haes/generate_data.py`

## Relevant Publication
If you use HA-ES in scientific publications, we would appreciate citations.

Maier, J., Möller, F., & Purucker, L. (2024). Hardware Aware Ensemble Selection for Balancing Predictive Accuracy and Cost. Paper presented at the Third International Conference on Automated Machine Learning (AutoML 2024) Workshop. arXiv. https://arxiv.org/abs/2408.02280


I have also written my Master's thesis on this topic: Hardware-Aware Ensemble Selection for Balancing Predictive Accuracy and Operational Costs in AutoML Systems
The thesis goes into much more detail and introduces some new methods. I would be happy to provide it to anyone interested. 
