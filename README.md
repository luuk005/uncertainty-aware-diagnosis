# uncertainty-aware-diagnosis

Uncertainty-Aware Medical Diagnosis Classification

Part of Luuk Jacobs' Master Thesis

To Do:
[] cross-validation MLP fit
[] Calibration wrapper / method to MLP class
[] improve calibration & hyperparameter tuning --> set up propper gridsearch pipeline
[] abstaining --> start with threshold on predicted_proba, then DAC
[] real data
[] eval on different levels of noise & models (uncalib, calib, abstaining-naive, DAC)
[] retraining strategies
[] custom class loss
[] custom noise-aware loss
[] federated setting (20% global class imbalance)
[] include text embedings | extract top key words, load in as embedings

### abstaining

1. [Deep Abstaining Classifier](https://github.com/thulas/dac-label-noise). use DAC to identify label noise, eliminate train samples that are abstained, retrain on leaner set using regular cross-entropy loss. (search Generalized Cross-entropy loss; zhang et al nips 2018) (you can stabilize abstention at known noise rate)
2. Selective Classification (Reject-Option Learning): learns a selection function (predict or abstain), with a single objective that trades off: classification loss on non-rejected examples and a penalty on the overall rejection (coverage) [see chat log](https://chatgpt.com/share/6808e212-b050-800d-a0f6-45397d4cf1a8)
