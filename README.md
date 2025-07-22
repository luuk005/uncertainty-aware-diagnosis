# Uncertainty-Aware Medical Diagnosis Classification

**Calibration experiment in noisy, extreme multi-class data from federated sources**

Qualitive error analysis investigating the question: _How does limited, noisy, and federated training data constrain the achievable calibration performance of extreme multi-class classification?_

Part of Luuk Jacobs' Master Thesis (2025)

## Relevant documents

`calibration_demo.ipynb` demonstrates how the code is used to train and evaluate different calibrators and more methods for dealing with noisy multi-class training data. Moreover it explains the design decisions and background info. (See "Get started" below to setup the demo).

This project consists of 3 packages:

### (1) `uncertainty_aware_diagnosis`

This folder defines the pytorch implementation of an uncertainty-aware Multi Layer Percepton (MLP) that aims to predict ICD-10 codes of hospital day-admissions (a noisy extreme multi-class classification problem).

- `mlp_classifier.py` contains the MLP class that is used as baseline model and the Platt's and Temperature scaling class that is used for calibration.
- `stratified_data_split.py` contains the function to creat the train, calibration, and test split.
- `load_icd10_data.py` contains the class to load and preprocess the dataset.
- `noise_robustness_improvements.py` defines a pytorch implementation of the `Symmetric Cross Entropy Loss` and `Monte Carlo Dropout`.
- `curriculum_aware_dataloader.py` defines a uncertainty-aware (pytorch) dataloader, that schedules the trainingdata according the obtained uncertainty. It also defines a function to visualise the curriculum sheduling during model training.
- The `calibration_demo` notebook (in the main project folder), shows how the code is implemented and how the experiments can be recreated.

### (2) `calibrators`

In this folder you find the implementations of different calibrators methods, together with an evaluation design that computes and compares calibration and performance metrics of the different methods and visualises it using the reliability diagram.

- `temperature.py` implements a (pytorch) temperature scaling calibrator: a reliable method that uses fits a single parameter, at the cost of flexibility.
- `dirichlet.py` implements a (pytorch) dirichlet calibrator: which is intentionally designed for multi-class classification and provides more flexibility than temperature scaling. It can improve performance (were temperature cannot) but might not achieve as much calibration.
- `evaluator.py` defines the `Evaluator` class that can compute and compare the following for the different calibration methods:
  - Performance metrics (larger -> better):
    - Macro-F1 (per-class balanced performance)
    - Macro Recall (classwise sensitivity)
  - Calibration metrics (smaller -> better):
    - Confidence Expected Calibration Error (detects global miscalibration; i.e.under/over confidence)
    - Classwise Expected Calibration Error (detects imbalanced calibration)
    - Barrier score loss (combines confidence and correctness)
  - Reliability diagram: plots the confidence of the winning class against the observed distribution to show how well-calibrated the predictioins are.
- `base.py` defines an abstract base for implementing the different calibrators in sklearn style.

### (3) `synthetic_admissions`

- Generates representative synthetic data for demo purposes (which can obtain comparable results), as the real hospital data is ofcourse strictly private.
- `synthetic_admissions_preprocessing.ipynb` shows you how to obtain the synthetic data and explains some details.

## Get started

1. Setup environment

```
# Clone the repository
git clone https://github.com/luuk005/uncertainty-aware-diagnosis

# setup env
uv sync
# activate env on windows
.venv\Scripts\activate
# activate on linux or mac
source .venv/bin/activate
```

2. Setup noisy multi-class data

- Run `synthetic_admissions_preprocessing.ipynb` to generate synthetic data that simulates hospital admission data from multiple hospitals (extreme multiclass data with heterogenious label noise and inter-hospital differences in class distribution)

3. Run the calibration demo!

- You can now run `calibration_demo.ipynb` which demonstrates how the code works, acompanied by explainations about the methods used and their underlying design decisions.

## Future work & Recommendations
1. Use Symmetric Cross Entropy loss (SCE; see [paper](https://arxiv.org/pdf/1908.06112)), it penalizes model overconfidence and is therefore more effective in learning from noisy training data. The `SymmetricCrossEntropyLoss` class is a pytorch implementation defined in `uncertainty-aware-diagnosis.noise_robustness_improvements.py`
2. Apply model confidence calibration to make the model estimates more aligned to reality. For ICD-10 diagnosis classification (limited, noisy, multi-class data) use Temperature scaling as a reliable baseline (improves calibration but ensures equal performance), and test if Dirichlet calibration can improve both calibarion and performance in your case. See for the implementation and evaluation methods the notebook: `calibarion_demo.ipynb`.
