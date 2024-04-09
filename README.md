# HA-ES: Hardware-Aware Ensemble Selection
One of the concerns with currently existing ensemble selection algorithms is the size of the ensemble, which can affect the inference speed of the trained model. To address this issue, we introduce our novel approach: hardware-aware ensemble selection (HA-ES), which focuses on finding a balance in the performance and complexity trade-off inherent in ensembling.

To evaluate our approach and compare it to the existing algorithms, we use TabRepo, which provides prediction probabilities for over 100 ML problems. We use this data to efficiently evaluate and compare the ensemble selection techniques.


## Set-Up
This set-up guide expects a Linux system.

### (Optional) Create venv
It's good practice to use a virtual environment. This isolates your project dependencies from global Python installations. This is how you create a virtual environment in your project directory:
- `python3 -m venv venv`
To activate it use:
- `source venv/bin/activate`


### Dependencies
From the project root run the following commands to install the dependencies (`-e` to automatically install changes made to the code of the dependencies)
- `python3 -m pip install -e extern/tabrepo`
- `python3 -m pip install -e extern/phem`

### Run test
To run the current state of development use
- `python3 haes/main.py`