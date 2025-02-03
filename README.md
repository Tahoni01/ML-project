ML Project 2024
Authors: Riccardo La Rosa, Sean Campailla and coordinated by prof. Alessio Micheli from University of Pisa.

The project
We had 4 tasks with a Dataset for each one:

3 of them were Binary Classification problems, as boolean expressions (Monks datasets),
1 of them was a Regression problem (CUP dataset).

Implementation
We’ve compared one Neural Networks in Pytorch library and one SVM using Scikit-Learn library, to see differences both on models and tools.

For an effective comparison, we used the specific provided functions of each library without mixing them, when possible. We made Model Selection and Model Assessment by using Grid-Search for SVM and Random Search for Pytorch, and both K-fold Cross-Validation before testing all models, on an Internal Test set for CUP, and on the given Test set  files (for Monks).

Regression problem (CUP)
The selected model for the CUP is made with both Pytorch and SVM. We focused on various levels of abstraction, flexibility and customizability offered by the different tools we investigated.

Why we have chosen these frameworks?
Regarding the Neural Networks, we utilized:
-PyTorch for its greater flexibility, customizability and makes you see the various stages of a neural network in real time. 
Furthermore, regaring SVM's models, we used:
-Sklearn for its semplicity and fast processing campability
