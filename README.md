# Fraud-Detection-in-Online-transaction
 Enhance online transaction security using machine learning on a Kaggle dataset. Employ Neural Networks, Random Forests, SVM, Logistic Regression to detect fraud patterns. Aim: fortify financial institutions against online fraud risks.

Fraud Detection in Online Transactions

Abstract: 
This report delves into the intricate landscape of fraud detection within online transactions using machine learning algorithms. Leveraging a dataset sourced from Kaggle the project meticulously navigates through data pre-processing, model development, and comparison. The overarching goal is to fortify the security of online financial transactions and minimize risks associated with fraudulent activities.

Introduction:
Online fraud poses a significant threat to financial institutions, necessitating the development of robust fraud detection models. This project explores the application of machine learning algorithms to address this challenge. By leveraging a dataset of online transactions, the goal is to enhance security and minimize financial losses. After meticulous data cleaning, machine learning algorithms, including the Neural network, Random Forest, support vector classifier, logistic regression are applied for fraud detection.

Objective:
The primary objective of the project is to fortify the security of online financial transactions and minimize risks associated with fraudulent activities using machine learning algorithms. The project focuses on leveraging a dataset sourced from Kaggle, comprising details of online transactions, to develop robust fraud detection models. The aim is to employ various machine learning algorithms, including Neural Network, Random Forest, Support Vector Classifier, and Logistic Regression, to analyze patterns and detect fraudulent transactions. The project encompasses data pre-processing, model development, and model comparison, ultimately emphasizing the importance of machine learning in enhancing the security of online transactions by addressing the challenges posed by financial fraud. The comprehensive analysis involves exploring the intricacies of fraud detection methodologies, such as combining supervised and unsupervised techniques, efficiency through parameter leveraging, predictive analytics, real-time transaction monitoring, baseline profiling, anomaly detection, application of unsupervised machine learning in financial auditing, voice recognition for identity verification, biometric identification methods, geolocation tracking, and data enrichment. The ultimate goal is to provide actionable insights and solutions for financial institutions to mitigate the impact of fraudulent activities in the online transaction landscape.








Importance of machine learning in detecting the fraud in online bank transaction:

1.	Combining Supervised and Unsupervised Techniques: Machine learning research in fraud detection acknowledges the complexity of financial fraud and the need for a combination of supervised and unsupervised techniques to build models with adequate predictive power and accuracy.

2.	Efficiency through Parameter Leveraging: Machine learning models leverage tens of thousands of parameters, enhancing their efficiency in detecting subtle connections within data, connections that may elude human or expert system analysis.

3.	Predictive Analytics for Historical Trend Identification: Predictive analytics using machine learning algorithms enables the detection and prevention of financial fraud by analyzing historical data, identifying trends, and patterns connected to fraudulent transactions.

4.	Real-Time Transaction Monitoring: Machine learning facilitates real-time transaction monitoring, allowing for the immediate observation of financial transactions as they occur. This includes diverse operations such as credit card purchases, fund transfers, withdrawals, and deposits.

5.	Baseline Profiles for Fraud Detection: Machine learning algorithms within transaction monitoring establish baseline profiles for every company or account holder, considering factors such as transaction frequency, amounts, locations, and typical transaction times.

6.	Anomaly Detection for Unusual Behavior: Anomaly detection in machine learning models involves training to detect irregularities in transactional and operational data. Alerts are triggered when a transaction significantly deviates from established patterns or the customer's usual behavior.

7.	Application of Unsupervised Machine Learning in Financial Auditing: Unsupervised machine learning approaches, such as isolation forest and autoencoders, are applied to identify irregularities and enhance the effectiveness of financial audits by addressing issues associated with variations in journal entry size.

8.	Voice Recognition for Identity Verification: Machine learning-driven voice biometrics is utilized for identity verification, assessing vocal characteristics during the user authentication process, providing an additional layer of security.

9.	Biometric Identification Methods: Machine learning models are employed for biometric identification methods, including face, voice, and fingerprint recognition, enhancing security and convenience for users.

10.	Geolocation Tracking and Data Enrichment: Geolocation tracking, coupled with machine learning, is used to detect irregularities by cross-referencing transaction locations with past data. Additionally, data enrichment through supplementary sources like social media profiles and public records enhances the identification of fraudulent activity.
(Sapra, Y. (2023, November 14).)




About data:
Following are the details about the columns of the data.

Type: This categorical column indicates the type of the transaction, providing insights into the nature of financial activities. It classifies transactions into different categories, such as 'cash in,' 'cash out,' 'debit,' 'transfer,' or 'payment,' offering a comprehensive overview of the diverse transactions within the dataset.

Amount: The 'amount' column represents the monetary value associated with each transaction. It signifies the financial magnitude of the transaction, providing critical information for understanding the economic impact of individual transactions within the dataset.

NameOrig: This column identifies the customer initiating the transaction. It serves as a unique identifier for the originator of the financial activity, contributing to the traceability of transactions back to their source.

OldbalanceOrg: 'OldbalanceOrg' signifies the account balance before the transaction for the customer initiating the financial activity. It captures the initial financial state of the originating account, offering a reference point for understanding the changes brought about by the transaction.

NewbalanceOrig: Reflecting the account balance after the transaction for the customer initiating the financial activity, 'NewbalanceOrig' provides insight into the resulting financial state of the originating account. It is instrumental in tracking the impact of the transaction on the account balance.

NameDest: This column specifies the recipient of the transaction. It uniquely identifies the destination of the financial activity, facilitating a clear understanding of the flow of funds within the dataset.

OldbalanceDest: 'OldbalanceDest' represents the initial balance of the recipient account before the transaction. It is a crucial parameter for evaluating the financial state of the destination account prior to the transaction.

NewbalanceDest: Indicating the new balance of the recipient account after the transaction, 'NewbalanceDest' provides a comprehensive view of the resulting financial state of the destination account. It is essential for assessing the impact of the transaction on the recipient account balance.

isFraud: The binary 'isFraud' column serves as a crucial indicator, categorizing transactions as either fraudulent (1) or non-fraudulent (0). It is the key variable for the fraud detection task, enabling the classification of transactions based on their fraudulent nature.



Data cleaning: 

Duplicated Rows:
Identify and count duplicated rows in the dataset.
Remove duplicated rows to ensure each transaction entry is unique.

Transaction Type and Fraud Analysis:
Group the data by transaction type ('type').
Count the occurrences of fraudulent and non-fraudulent transactions within each transaction type.
This analysis provides insights into the distribution of fraud across different types of transactions, where we found out that fraud was in Cash out and Transfer transaction type.

Checking Null Values:
Examined the dataset for the presence of null values.
Ensured that the dataset does not contain missing values that could impact the analysis.
Removing Irrelevant Columns:

Removing the irrelevant column:
Excluded columns deemed irrelevant for subsequent analysis (e.g., 'step,' 'nameOrig,' 'nameDest,' 'isFlaggedFraud,' 'isFraud').
Focus on essential features for the development of machine learning models.

Creating Dummy Variables:
Converted categorical variables, such as the 'type' column, into dummy variables.
Dummy variables are binary indicators representing different categories and are useful for machine learning algorithms.

Standardizing the Input Data:
Standardize the numerical features of the dataset using techniques like z-score normalization. Standardization ensures that numerical variables are on a similar scale, preventing certain machine learning algorithms from being influenced by the magnitude of features.


These cleaning and pre-processing steps are essential to ensure the dataset is free from inconsistencies, duplicates, and irrelevant information. The transformation of categorical variables and standardization of numerical features contribute to creating a dataset suitable for the development of accurate machine learning models for fraud detection.

Visualization: 
![image](https://github.com/Sagarlimbachiya1993/Fraud-Detection-in-Online-transaction/assets/106364353/88eb0f12-3901-4129-a7ea-795b0cd66586)

 ![image](https://github.com/Sagarlimbachiya1993/Fraud-Detection-in-Online-transaction/assets/106364353/81ffc761-4475-4ccc-a5e5-ca3fdd4c1ea9)


 

As we can see there is the class imbalance problem in classification of the data, i.e., there are only 8213 0s out of the 108097 rows. so we need to do the oversampling, which will be done further in the process for that we will use SMOT method.


About Algorithms:
Neural network:
The neural network is trained on historical data to recognize patterns of online payment usage for individual consumers, incorporating various categories such as cardholder occupation, income, and purchase behaviour. The system uses prediction algorithms to classify transactions as fraudulent or genuine by comparing them to learned patterns. The neural network's ability to analyse and adapt to patterns allows it to detect potential fraud, particularly when unauthorized users deviate from the established patterns. The article highlights the neural network's real-value output between 0 and 1, with a threshold set to determine the likelihood of a transaction being illegal. The neural network's capacity to differentiate between legitimate variations and potentially fraudulent activities, such as large and rapid purchases, makes it a suitable technology for credit card fraud detection. (Patidar, R., & Sharma, L.)

Support vector machine:
SVC is a supervised learning algorithm that excels in binary classification tasks, making it well-suited for distinguishing between legitimate and fraudulent transactions. In the context of online bank transactions, where the dataset is characterized by a large number of features and patterns, SVC can effectively identify the boundaries between classes. SVC operates by finding the hyperplane & we are looking for the optimal separating hyperplane between the two classes by maximizing the margin between the classes’ closest points. This is particularly advantageous in fraud detection, where the distinction between normal and fraudulent transactions may not be straightforward. SVC can identify intricate patterns and non-linear relationships within the data, providing a robust solution for detecting subtle anomalies indicative of fraud.( Meyer, D., & Wien, F. T.)

Random Forest classifier: 
In the context of monitoring cardholder transaction behaviour, the planned system employs the Random Forest algorithm for classifying online transaction datasets. Random Forest, characterised as an algorithm for both classification and regression, consists of decision tree classifiers. Unlike individual decision trees, Random Forest addresses overfitting by randomly sampling subsets of the training set for each tree, leading to better generalisation. Notably, the algorithm is advantageous for large datasets with numerous features, as training is swift, and each tree is developed independently. The algorithm's resistance to overfitting, quick training, and ability to estimate generalization error contribute to its effectiveness. Random Forest selects the best feature from a random subset, improving model performance. In the realm of fraudulent activities detection, the proposed online transactions fraud detection system employs machine learning, specifically Random Forest, due to its efficiency and accuracy. The algorithm's ability to reduce correlation issues, de-correlate trees, and set stopping criteria for node splits makes it an advanced and effective choice for online transaction fraud detection, outperforming other machine learning algorithms.( Jonnalagadda, V., Gupta, P., & Sen, E.)

Logistic Regression:
Logistic Regression is suitable for online transaction fraud detection due to its effectiveness in binary classification tasks. It provides interpretable results, handles linear decision boundaries efficiently, and offers probabilistic outputs for nuanced decision-making. The model is robust to noise, scalable for large datasets, and allows for the identification of important features. While Logistic Regression is valuable for its simplicity and interpretability, more complex models may be considered for highly non-linear fraud patterns. The choice depends on specific data characteristics and detection goals.


Discussions: 

In this segment of the code, the journey into creating a robust fraud detection system unfolds. The initial step involves dividing the dataset into training and testing sets, a crucial practice for assessing the models' real-world performance. As the stage is set, three formidable machine learning models step into the scene: the Multi-Layer Perceptron (MLP) Classifier, Logistic Regression, Random forest classifier and the Support Vector Machine (SVM). Their capabilities are harnessed to discern patterns within the data, seeking to identify potential instances of fraud.

The SVM model, however, undergoes a meticulous process of hyperparameter tuning, utilizing the powerful GridSearchCV to select the most fitting kernel for the dataset. The Radial Basis Function (RBF) kernel emerges as the ideal parameter, guiding the SVM towards optimal performance. This careful selection process sets the stage for the subsequent use of the SVM in fraud detection.

To address the inherent imbalance between non-fraudulent and fraudulent transactions, a strategic move is made with the implementation of the Synthetic Minority Over-sampling Technique (SMOTE). This technique creatively augments the dataset, ensuring a balanced representation of both classes and fortifying the models against skewed predictions.

The oversampled training set, a testament to the rebalancing act performed by SMOTE, is visually inspected. Here, the intricate dance between non-fraudulent and fraudulent instances reveals a balanced ratio, laying the foundation for more accurate and fair model training.

A critical phase follows with the introduction of cross-validation, utilizing the oversampled/resampled training data. Each model undergoes a rigorous evaluation process, allowing for a nuanced understanding of their performance. The average scores across multiple folds are calculated, offering a comprehensive perspective on each model's ability to generalize to unseen data.

As the models are unleashed on the data, a clear victor emerges. The Random Forest Classifier, boasting a consortium of decision trees, outshines its counterparts with an impressive accuracy score of 99.7%. Classification reports are then generated, unveiling the precision, recall, and F1-score for each model, both in the training and testing sets. These reports, akin to a performance review, provide a detailed breakdown of the models' strengths and weaknesses.

The story concludes with a resounding endorsement for the Random Forest Classifier. Its ability to navigate the intricate landscape of the dataset and deliver accurate predictions positions it as the stalwart guardian against fraudulent activities. The findings from this code segment, woven into the broader narrative of the report, accentuate the importance of model selection in fortifying the security of online financial transactions.

Conclusion: 

In the final sections, the report draws meaningful conclusions from the findings, summarizing the key takeaways from the analysis. The overarching narrative underscores the significance of the Random Forest Classifier, which achieves a remarkable accuracy of more than 99 percent, signalling promising avenues for enhancing the security of online financial transactions through robust fraud detection systems.

Classification reports were generated for each model on both the training and test sets. The Random Forest Classifier demonstrated superior precision, recall, and F1-score, reaffirming its efficacy in fraud detection. 


References:
•	Patidar, R., & Sharma, L. (2011). Credit card fraud detection using neural network. International Journal of Soft Computing and Engineering (IJSCE), 1(32-38).
•	Meyer, D., & Wien, F. T. (2001). Support vector machines. R News, 1(3), 23-26.
•	Jonnalagadda, V., Gupta, P., & Sen, E. (2019). Credit card fraud detection using Random Forest Algorithm. International Journal of Advance Research, Ideas and Innovations in Technology, 5(2), 1-5.
•	Sapra, Y. (2023, November 14). How Machine Learning in Banking Helps in Fraud Detection. https://hashstudioz.com/blog/how-machine-learning-in-banking-helps-in-fraud-detection/

 
