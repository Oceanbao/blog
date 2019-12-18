---
title: "STATS"
date: 2019-12-018T15:52:43-05:00
showDate: true
draft: false
---

# INTRO

- Statistical Learning (aka Machine Learning) becoming distinct subset of Statistical Science focusing on **prediction and inference** 

- Inception from **Least Square** by Gauss et al, followed by **Fisher's Logistic Regression** and **Generalised Linear Model** in 1960s through 1980s

- **Reducible Error** vs **Irreducible Error**

  $$E(Y - \hat{Y})^2 = E[f(X) + \varepsilon - \hat{f}(X)]^2 = [f(X) - \hat{f}(X)]^2 + Var(\varepsilon)$$

  - $$\hat{f}$$ can be improved by model and data
  - $$Var(\varepsilon)$$ as variance of **all else** irreducible, unknown in practice

- **Parametric** method simplifies estimation of *arbitrary p-dimensional* $$f$$ forms to estimating set of **p + 1 coefficients**

- **LASSO** is more interpretable that **OLS** for only a small subset of predictors - **nonzero coefficient estimates** are related to response variable