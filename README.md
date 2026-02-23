# Deep-Learning-Final-project-Breast-Tumor-Classification

This repository contains scripts for classifying breast cancer histopathology images from the BreaKHis dataset. These scripts were excecuted on Run:ai GPU cluster.

Repository Contents:
BreakHis_binary_Baseline_1.py: Baseline model using a 1:1 binary classification approach.

BreakHis_binary_Baseline_2.py: Advanced baseline using Patient ID-based splitting to prevent data leakage and ensure images from the same patient aren't shared across sets.

BreakHis_grid_search.py: Grid search used (loss functions, Dropout, etc.).

requirements.txt: Python dependencies including TensorFlow, Scikit-learn etc.
