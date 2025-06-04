# Uncertainty-Aware Medical Diagnosis Classification

**Calibration experiment in noisy, extreme multi-class data from federated sources**

Qualitive error analysis investigating the question: _How does limited, noisy, and federated training data constrain the achievable calibration performance of extreme multi-class classification?_

Part of Luuk Jacobs' Master Thesis

## Relevant documents:

- The experiments were conducted in the `calibration_experiment` notebook.
- the `uncertainty_aware_diagnosis` package contains the imported functions
  - `mlp_classifier.py` contains the MLP class that is used as baseline model and the Platt's and Temperature scaling class that is used for calibration.
  - `stratified_data_split.py` contains the function to creat the train, calibration, and test split.
  - `load_icd10_data` contains the class to load and preprocess the dataset.

## Get started:

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
