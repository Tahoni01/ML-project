ML Project 2024
Authors
Riccardo La Rosa, Sean Campailla
Supervised by Prof. Alessio Micheli – University of Pisa

Project Overview
This project focuses on Machine Learning by tackling four different tasks, each with its own dataset:

3 Binary Classification Problems: Solving Boolean expressions using the Monks datasets.
1 Regression Problem: Addressing a regression task with the CUP dataset.
Implementation
We implemented and compared two different machine learning approaches:

Neural Networks using PyTorch
Support Vector Machines (SVMs) using Scikit-Learn
The goal was to analyze differences both in model performance and in the tools used.

To ensure a fair and effective comparison, we:
✔ Used the specific built-in functions of each library, avoiding unnecessary mixing.
✔ Applied Grid Search (SVM) and Random Search (PyTorch) for hyperparameter tuning.
✔ Conducted K-Fold Cross-Validation before testing the models.
✔ Evaluated performance on:

An internal test set for the CUP dataset.
The provided test files for the Monks datasets.
Regression Task (CUP Dataset)
For the CUP dataset, we implemented models using both PyTorch and SVM.
The study focused on different levels of abstraction, flexibility, and customizability offered by these tools.

Why These Frameworks?
🔹 Neural Networks (PyTorch)
Chosen for its flexibility, customizability, and real-time visualization of neural network operations.
🔹 Support Vector Machines (Scikit-Learn)
Selected for its simplicity and fast processing capabilities, making it an ideal tool for quick experimentation.

