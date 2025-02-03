🚀 ML Project 2024

👨‍💻 Authors

Riccardo La Rosa

Sean Campailla📌 Supervised by Prof. Alessio Micheli – University of Pisa

📌 Project Overview

This project focuses on Machine Learning by tackling four different tasks, each with its own dataset:

🔹 3 Binary Classification Problems – Boolean expressions using the Monks datasets.🔹 1 Regression Problem – Addressing a complex regression task with the CUP dataset.

⚙️ Implementation

We implemented and compared two different machine learning approaches:

🔹 Neural Networks using 🧠 PyTorch🔹 Support Vector Machines (SVMs) using ⚡ Scikit-Learn

🔎 Evaluation & Model Selection

To ensure a fair and effective comparison, we applied:✔ Hyperparameter Optimization:

Grid Search (SVM)

Random Search (PyTorch)✔ K-Fold Cross-Validation for reliable performance assessment.✔ Model Testing on:

An internal test set (CUP dataset).

The provided test files (Monks datasets).

📊 Regression Task (CUP Dataset)

For the CUP dataset, we implemented models using both PyTorch and SVM.Our study focused on the different levels of abstraction, flexibility, and customizability offered by these tools.

❓ Why These Frameworks?

🧠 Neural Networks with PyTorch

✅ High flexibility & customizability.✅ Provides real-time insights into each stage of a neural network.✅ Ideal for complex deep learning architectures.

⚡ Support Vector Machines with Scikit-Learn

✅ Extremely fast and easy to use.✅ Well-optimized for small to medium-sized datasets.✅ Great for quick experimentation and benchmarking.

📂 Repository Structure

📂 ML-Project-2024/
│── 📁 data/             # Contains dataset files (Monks & CUP)
│── 📁 models/           # Implementations of NN (PyTorch) & SVM (Scikit-Learn)
│── 📁 results/          # Training logs, plots, and final outputs
│── 📜 preprocess.py     # Data loading & preprocessing functions
│── 📜 main.py           # Main script to execute experiments
│── 📜 README.md         # Project documentation (this file)

🚀 How to Run the Code

1️⃣ Clone the Repository

git clone https://github.com/yourusername/ML-Project-2024.git
cd ML-Project-2024

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Run the Main Script

python main.py

📌 Results & Insights

🔹 Results from classification models (Monks datasets) and regression models (CUP dataset) are available in the results/ folder.🔹 Learning curves and model comparisons can be visualized using the provided plotting functions.

📈 For detailed performance analysis, check the Jupyter Notebook (coming soon!)

📧 Contact & Contributions

For any questions or contributions, feel free to open an issue or submit a pull request!

📩 Contact: riccardo.la.rosa@email.com | sean.campailla@email.com🌍 University of Pisa
