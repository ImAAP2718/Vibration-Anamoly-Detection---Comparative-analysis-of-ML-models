# Vibration-Anamoly-Detection---Comparative-analysis-of-ML-models
README – Induction Motor Imbalance Fault Detection
This repository contains two Python scripts that implement vibration-based fault detection for induction motors.
Both scripts perform data loading → downsampling → FFT feature extraction → class balancing → model training → internal & external evaluation, but at different levels of detail.
Files in this Project
1.	updated_30gcase_as_external_data_for_testing  is the main analysis script for this project. It loads the full vibration dataset (normal and all imbalance levels), performs downsampling and FFT based feature extraction, balances the training data, and then trains four models: SVM, kNN, DNN, and Gaussian Naive Bayes. For each model it computes detailed metrics (accuracy, precision, recall, F1 score), generates internal and external (30 g) confusion matrices, performs misclassification analysis, and produces comparison plots across all four metrics. This script is intended for deep analysis and for generating results and figures suitable for use in your report.
2.	Data Analysis and Fault detection file,  is a lighter version focused on quick evaluation. It follows the same general pipeline (data loading, preprocessing, model training) but reports a more basic set of metrics and uses standard confusion matrices and simpler plots, typically centered on accuracy. It is useful when you want to verify that the pipeline is working and obtain a rapid performance check without generating the full suite of diagnostic figures and misclassification summaries.

Dependencies
 Install the required libraries manually using
 pip install numpy pandas matplotlib seaborn scipy scikit-learn tqdm tensorflow. 
These packages cover numerical operations (numpy), dataframes and CSV loading (pandas), plotting (matplotlib, seaborn), FFT and autocorrelation (scipy), progress bars (tqdm), classical machine learning models and preprocessing (scikit-learn), and the deep neural network implementation (tensorflow / keras), while (glob) from the Python standard library is used to read multiple CSV files.

Expected Outputs
For both scripts, the main outputs are a mix of console text and plots that summarize how well each model detects imbalance. After running a script, the terminal will first show data loading information (number of files per class and shapes), followed by class counts before and after balancing. Then, for each model (SVM, kNN, DNN, GaussianNB), you will see its internal and external (30 g) test accuracies, along with full classification reports listing precision, recall, F1 score, and support for each class. Misclassification summaries will print the number of incorrect predictions, the indices of a few misclassified samples, and their true vs predicted labels so you can inspect difficult cases.
Graphical outputs include several figures that pop up as separate windows. You will see 2×2 confusion matrices for the internal test set for all models, plus a custom 1×2 confusion matrix for the external 30 g data that explicitly shows how many imbalance samples were correctly detected and how many were misclassified as normal. For the DNN, there are training and validation accuracy and loss curves over epochs to assess convergence and overfitting. Finally, comparison bar charts summarize model performance: in the full script, there are four sets of plots comparing accuracy, precision, recall, and F1 score for all models on internal vs external tests, while the simpler script focuses mainly on accuracy (and possibly a reduced set of metrics) for a quicker visual comparison.
Differences in the files
The script updated_30gcase_as_external_data_for_testing,  provides a more detailed analysis than Data Analysis and Fault detection file, .py. The updated script includes full misclassification analysis, computing and reporting precision, recall, and F1-score for all four models, and generates a custom confusion matrix for the 30 g external test that explicitly shows both correctly detected imbalance samples and those misclassified as normal. It also produces a set of model comparison plots covering four metrics (accuracy, precision, recall, and F1-score), making it suitable for in depth analysis and report writing. In contrast, Data Analysis and Fault detection file, .py is a lighter script focused on quick evaluation: it reports only basic metrics and standard confusion matrices and provides accuracy based comparison plots, but omits the extended error analysis and detailed metric visualizations.

Notes
•	The external 30g dataset is never used in training for the updated file; it is only for evaluating model generalization.
•	Both scripts assume all vibration signals have consistent structure (same number of columns).
•	Downsampling step is fixed at 5000 samples per block.
•	Ensure the location of the data files is updated in the code before running. 

