k-NN (k-Nearest Neighbors) Algorithm Implementation



Project Overview
This project implements the k-Nearest Neighbors (k-NN) algorithm from scratch, which is a simple and powerful supervised machine learning algorithm used for classification tasks. The goal of the project is to apply k-NN on the Wine dataset and explore the effect of different values of k and distance metrics on the model's performance.
The project includes:
•	Implementation of the k-NN algorithm with two distance metrics: Euclidean and Manhattan.
•	Comparison of the performance of the model with varying values of k (1, 3, 5, 7, 9).
•	Visualization of accuracy vs. k-values for both Euclidean and Manhattan distance metrics.
•	Performance evaluation using confusion matrices and classification reports.




Requirements




Before running the code, you will need to install the following Python libraries:
•	numpy: for numerical computations
•	pandas: for data manipulation and analysis
•	matplotlib: for plotting graphs
•	seaborn: for statistical data visualization
•	sklearn: for utilities like dataset loading, model evaluation, etc.




Dataset



The dataset used in this project is the Wine dataset, which contains data about different wine varieties. The dataset consists of 13 features (such as alcohol content, phenols, and acidity) and 3 classes corresponding to 3 different wine types.

The dataset is included as wine.data in the project directory.



Instructions to Run the Code
1.	Clone the repository or download the project files to your local machine.
2.	Prepare your environment:
o	Ensure that Python and required libraries are installed (as mentioned in the requirements).
3.	Run the Jupyter Notebook:
o	Open the analysis.ipynb notebook using Jupyter Lab or Jupyter Notebook.
4.	Execute the code:
o	The notebook is structured with step-by-step instructions and code blocks. Simply run each cell to load the dataset, train the model, and generate the results.
5.	Review the results:
o	The code will output the confusion matrix, classification report, and accuracy vs. k-value graph comparing Euclidean and Manhattan metrics.
6.	Experiment with different values of k and distance metrics to explore the impact on model performance.

