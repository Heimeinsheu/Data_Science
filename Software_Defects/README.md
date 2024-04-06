# Using Machine Learning Algorithms to Estimate Software Defects


## Abstract

<p style="text-align:justify" >
Software Engineering is a comprehensive domain since it requires a tight communication between system stake holders and delivering the system to be developed within a determinate time and a limited budget. Delivering the customer requirements include procuring high performance by minimizing the system. Thanks to effective prediction of system defects on the front line of the project life cycle, the project’s resources and the effort or the software developers can be allocated more efficiently for system development and quality assurance activities. The main aim of this paper is to evaluate the capability of machine learning algorithms in software defect prediction and find the best category while comparing five machine learning algorithms within the context of five NASA datasets obtained from public PROMISE repository and five more datasets from other sources.
</p>

---
---

## Introduction
<p style="text-align:justify">
Developing a software system is an arduous process which contains planning, analysis, design, implementation, testing, integration and maintenance. A software engineer is expected to develop a software system on time and within limited the budget which are determined during the planning phase. During the development process, there can be some defects such as improper design, poor functional logic, improper data handling, wrong coding, etc. and these defects may cause errors which lead to rework, increases in development and maintenance costs decrease in customer satisfaction. A defect management approach should be applied in order to improve software quality by tracking of these defects. In this approach, defects are categorized depending on the severity and corrective and preventive actions are taken as per the severity deﬁned. Studies have shown that ’defect prevention’ strategies on behalf of ’defect detection’ strategies are used in current methods. Using defect prevention strategies to reduce defects generating during the software development the process is a costly job. It requires more effort and leads to increases in project costs. Accordingly, detecting defects in the software on the front line of the project life cycle is crucial. The implementation of machine learning algorithms which is the binary prediction model enables identify defect- prone modules in the software system before a failure occurs during development process. In this research, our aim is to evaluate the software defect prediction performance of seven machine learning algorithms by utilizing quality metrics; accuracy, precision, recall, F-measure associated with defects as an independent variable and ﬁnd the best category while comparing software defect prediction performance of these machine learning algorithms within the context of four NASA datasets obtained from public PROMISE repository. The selected machine learning algorithms for comparison are used for supervised learning to solve classiﬁcation problems. They are three tree-structured classiﬁer techniques: (i) Bagging, (ii) Random Forests (RF) and (iii) Decision Tree; One Neural networks techniques: (i) Multilayer Perceptron (MLP); and one discriminative classiﬁer Support Vector Machine (SVM). The remainder of the paper is organized as follows: Section 2 brieﬂy describes the related work, while Section 3 describes the experimental methodology in detail. Section 4 contains the conclusion of the experimental study and underlined some possible future research directions.
</p>

---
---

## Related Work
<p style="text-align:justify">
There are a great variety of studies which have developed and applied statistical and machine learning based models for defect prediction in software systems. Basili et al. (1996) [1] have used logistic regression in order to examine what the effect of the suite of object-oriented design metrics is on the prediction of fault-prone classes. Khoshgoftaar et al. (1997) [7] have used the neural network in order to classify the mod- ules of large telecommunication systems as fault-prone or not and compared it with a non-parametric discriminant model. The results of their study have shown that compared to the non-parametric discriminant model, the predictive accuracy of the neural network model had a better result. Then in 2002 [6], they made a case study by using regression trees to classify fault-prone modules of enormous telecommunication systems. Fenton et al. (2002) [4] have used Bayesian Belief Network in order to identify software defects. However, this machine learning algorithm has lots of limitations which have been recognized by Weaver(2003) [14] and Ma et al. (2007) [9]. Guo et al. (2004) [5] have applied Random Forest algorithm on software defect dataset introduced by NASA to predict fault-prone modules of software systems and compared their model with some statistical and machine learning models. The result of this comparison has shown that compared to other methods, the random forest algorithm has given better predictive accuracy. Ceylan et al. (2006) [2] have proposed a model which uses three machine learning algorithms that are Decision Tree, Multilayer Perceptron and Radial Basis Functions in order to identify the impact of this model to predict defects on different software metric datasets obtained from the real*life projects of three big-size software companies in Turkey. The results have shown that all of the machine learning algorithms had similar results which have enabled to predict potentially defective software and take actions to correct them. Elish et al. (2008) [3] have investigated the impact of Support Vector Machines on four NASA datasets to predict defect-proneness of software systems and compared the prediction performance of SVM against eight statistical and machine learning models. The results have indicated that the prediction performance of SVM has been much better than others. Kim et al. (2011) [8] have investigated the impact of the noise on defect prediction to cope with the noise in defect data by using a noise detection and elimination algorithm. The results of the study have presented that noisy instances could be predicted with reasonable accuracy and applying elimination has improved the defect prediction accuracy. Wang at all. (2013) [13] have investigated re-sampling techniques, ensemble algorithms and threshold moving as class imbalance learning methods for software defect prediction. They have used different methods and among them, AdaBoost.NC had better defect prediction performance. They have also improved the effectiveness and efﬁciency of AdaBoost.NC by using a dynamic version of it. Ren at al. (2014) [11] have proposed a model to solve the class imbalance problem which causes a reduction in the performance of defect prediction. The Gaussian function has been used as kernel function for both the Asymmetric Kernel Partial Least Squares Classiﬁer (AKPLSC) and Asymmetric Kernel Principal Component Analysis Classiﬁer (AKPCAC) and NASA and SOFTLAB datasets have been used for experiments. The results have shown that the AKPLSC had better impact on retrieving the loss caused by class imbalance and the AKPCAC had better performance to predict defects on imbalanced datasets. There is also a systematic review study conducted by Malhotra to review the machine learning algorithms for software fault prediction.
</p>

---
---

## Experimental Methodology
### 1. Datasets and Data Pre-Processing
<p style="text-align:justify">
The datasets which are available from the public PROMISE repository and used for this task are detailed in Figure I and the other datasets are detailed in Figure II. These datasets have different number of instances. The dataset with the most data in terms of the number of instances is JM1 with 10879 instances. Data sets of different sizes have been selected to demonstrate the effect of data size on accuracy. In Table I, each dataset explained with language, number of attributes, number of instances, percentage of defective modules and description. The number of attributes is 22 for KC1, KC2, CM1, PC1 and JM1 datasets and 30 for AR1, AR3, AR4, AR5 and AR6 datasets. Further we check for Null values
</p>

``` py
kc1_df.isnull().sum()
```

<p style="text-align:justify">
Checking type of each attribute in dataset. Converting
attribute of following dataset into required datatype. While 
converting attribute having object type into numeric type. 
Encounter many rows contain “?” value in respective attribute 
having object datatype. Instead of null “?” was used. Scaling 
datasets.
</p>

Attribute information is shown in Figure I And Figure II

-----------------------------------------------------------

<p style="text-align:justify">
We noted a huge class imbalance issue with the available 
datasets (faulty, non-faulty) as revealed in the figures below(Fig 
III to Fig X) which can cause high bias and lead to wrong 
prediction. We have used several methods to counter class
imbalance.
</p>

<p style="text-align:justify">

**K-fold Cross-Validation (CV)** model is employed for each learning algorithm to model validation. The k value is determined as 10 in this experiment. Since the number of samples in the used datasets are equal to 10, the data is divided into 10 folds. That means k-1 objects in the dataset are used as training samples and one object is used as test sample in the each iteration. That is, every data fold is used as a validation set exactly once and falls into a training set k-1 times. Then the average error across all k trials which is equal to the number of samples in the dataset is computed.

</p>

<p style="text-align:justify">

**SMOTE (Synthetic Minority Oversampling Technique)** is 
a data augmentation technique that helps to address the issue 
of imbalanced classes in machine learning datasets. It works 
by creating synthetic data points that are similar to the minority 
class data points, in order to balance the dataset and improve 
the performance of machine learning models.

</p>

<p style="text-align:justify">

**Stratified Sampling** is a sampling method that reduces the 
sampling error in cases where the population can be partitioned into subgroups. We perform Stratified Sampling by dividing the population into homogeneous subgroups, called strata, and then applying Simple Random Sampling within each subgroup.

</p>

<p style="text-align:justify">

**Shuffle Split** Unlike K-Fold, Shuffle Split leaves out a 
percentage of the data, not to be used in the train or 
validation sets. To do so we must decide what the train and 
test sizes are, as well as the number of splits.

</p>

---

### 2. Learning Algorithms

<p style="text-align:justify">
In this experiment, we have used for defect prediction in
software systems. They categorized the machine learning
algorithms based on distinct learners such as Ensemble
Learners, Neural Networks and SVM.According to these 
categories, we selected five different machine learning
algorithms to estimate software defect. Each algorithm is
detailed below in Table I.
</p>

#### 1. Ensemble Learners

<p style="text-align:justify">

- **Bagging:** This algorithm which is introduced by Leo Breiman and also called Bootstrap Aggregation is one of the ensemble methods. In this approach, N sub-samples of data from the training sample are created and the predictive model is trained by using these subset data. Sub-samples are chosen randomly with replacement. As a result, the ﬁnal model is an ensemble of different models.

- **Random Forest:** Random Forest algorithms which also called random decision forest is an ensemble tree-based learning algorithm. It makes a prediction over individual trees and selects the best vote of all predicted classes over trees to reduce overﬁtting and improve generaliza tion accuracy. It is also the most ﬂexible and easy to use for both classiﬁcation and regression. 

- **Decision Tree:** Decision Tree algorithm is a supervised learning technique that can be used for both classification and regression problems. It is a tree-structured classifier where internal nodes represent the features of a dataset, branches represent the decision rules, and each leaf node represents the outcome.

</p>

#### 2. Neural Network

<p style="text-align:justify">

- **Simple Perceptron:** The Simple Perceptron is a basic type of artificial neural network that consists of a single layer of artificial neurons, also known as perceptron .The Simple Perceptron is a binary classifier that learns from labeled training data to make predictions. 

- **Multilayer Perceptron:** Multilayer Perceptron which is one of the types of Neural Networks comprises of one input layer, one output layer and at least one or more hidden layers. This algorithm transfers the data from the input layer to the output layer, which is called     feed forward.

- **Multilayer Neural Network + Permutation:** Similiar to MLNN, layers and neurons in each layer are arranged in permutation manner.

</p>

#### 3. Support Vector Machines:

<p style="text-align:justify">

**Support vector machine (SVM)** is a supervised machine learning method capable of both classification and regression. It is one of the most effective and simple methods used in classification. For classification, it is possible to separate two groups by drawing decision boundaries between two classes of data points in a hyperplane. The main objective of this algorithm is to find optimal hyperplane.

</p>

---

### 3. Evaluation Metrics

<p style="text-align:justify">
To evaluate learning algorithms which are stated above, commonly used evaluation metrics are used such as accuracy, precision, recall, F-measure. The performance of the model of each algorithm is evaluated by using the confusion matrix which is called as an error matrix and is a summary of prediction results on a classification problem. Evaluation of model is the most important for classification problem where the output can be of two or more types of classes and the confusion matrix is one of the most commonly used and easiest metrics for determining the accuracy of the model. It has True Positive (TP), True Negative (TN), False Positive (FP) and False Negative (FN) values.
• Positive (P) : Observation is positive (for example: is an
defective).
• Negative (N) : Observation is not positive (for example:
is not an defective).
• True Positive (TP) : The model has estimated true and
the test data is true.
• False Negative (FN) : The model has estimated false and
the test data is true.
• True Negative (TN) : The model has estimated false and
the test data is false.
• False Positive (FP) : The model has estimated true and
the test data is false.
 

1) Accuracy: Accuracy which is called classification rate
is given by the following relation:
Accuracy =(TP + TN)/(TP + TN + FP + FN)	(1)
2) Recall: To get the value of Recall, correctly predicted
positive observations is divided by the all observations in
actual class and it can be defined as below:
Recall =TP/(TP + FN)			(3)
3) Precision: Precision is the ratio of the total number
of correctly classified positive examples to the number of
predicted positive examples. As shown in Equation 4, As
decreases the value of FP, precision increases and it indicates
an example labeled as positive is indeed positive.
Precision = TP/(TP + FP)			(4)



4) F-measure: Unlike recall and precision, this metric
takes into account both false positives(FP) and false negatives(
FN). F-measure is the weighted harmonic mean of the
precision and recall of the test. The equation of this metric
is shown in Equation 5.
Precision = (2 ∗ Recall ∗ Precision)/(Recall + Precision)       (5)

</p>

---
