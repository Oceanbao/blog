---
title: "ML"
date: 2019-12-018T15:52:43-05:00
showDate: true
draft: false
---

### Class Imbalance

An entry from the *Encyclopedia of Machine Learning* (https://cling.csd.uwo.ca/papers/cost_sensitive.pdf) helpfully explains that what gets called "the class imbalance problem" is better understood as three separate problems:

```
 (1) assuming that an accuracy metric is appropriate when it is not

 (2) assuming that the test distribution matches the training 
     distribution when it does not

 (3) assuming that you have enough minority class data when you do not
```

The authors explain:

> The class imbalanced datasets occurs in many real-world applications where the class distributions of data are highly imbalanced. Again, without loss of generality, we assume that the minority or rare class is the positive class, and the majority class is the negative class. Often the minority class is very small, such as 1%of the dataset. If we apply most traditional (cost-insensitive) classifiers on the dataset, they will likely to predict everything as negative (the majority class). This was often regarded as a problem in learning from highly imbalanced datasets.
>
> However, as pointed out by (Provost, 2000), two fundamental assumptions are often made in the traditional cost-insensitive classifiers. The first is that the goal of the classifiers is to maximize the accuracy (or minimize the error rate); the second is that the class distribution of the training and test datasets is the same. Under these two assumptions, predicting everything as negative for a highly imbalanced dataset is often the right thing to do. (Drummond and Holte, 2005) show that it is usually very difficult to outperform this simple classifier in this situation.
>
> Thus, the imbalanced class problem becomes meaningful only if one or both of the two assumptions above are not true; that is, if the cost of different types of error (false positive and false negative in the binary classification) is not the same, or if the class distribution in the test data is different from that of the training data. The first case can be dealt with effectively using methods in cost-sensitive meta-learning.
>
> In the case when the misclassification cost is not equal, it is usually more expensive to misclassify a minority (positive) example into the majority (negative) class, than a majority example into the minority class (otherwise it is more plausible to predict everything as negative). That is, FN > FP. Thus, given the values of FN and FP, a variety of cost-sensitive meta-learning methods can be, and have been, used to solve the class imbalance problem (Ling and Li, 1998; Japkowicz and Stephen, 2002). If the values of FN and FP are not unknown explicitly, FN and FP can be assigned to be proportional to p(-):p(+) (Japkowicz and Stephen, 2002).
>
> In case the class distributions of training and test datasets are different (for example, if the training data is highly imbalanced but the test data is more balanced), an obvious approach is to sample the training data such that its class distribution is the same as the test data (by oversampling the minority class and/or undersampling the majority class)(Provost, 2000).
>
> Note that sometimes the number of examples of the minority class is too small for classifiers to learn adequately. This is the problem of insufficient (small) training data, different from that of the imbalanced datasets.

Thus, as Murphy implies, there is nothing inherently problematic about using imbalanced classes, provided you avoid these three mistakes. Models that yield posterior probabilities make it easier to avoid error (1) than do discriminant models like SVM because they enable you to separate inference from decision-making. (See Bishop's section 1.5.4 *Inference and Decision* for further discussion of that last point.)

https://www.svds.com/learning-imbalanced-classes/