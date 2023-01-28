# S3-method-shallow-neural-network-and-C-implementation

#### BIOST 561: Computational Skills for Biostatistics
#### Instructor: Eardi Lila

## Problem 1
In this problem, you will refactor the shallow neural network code for binary classification provided in class
and use the S3 object-oriented system.

#### 1.1 To this end, you will define a new S3 class named shallow_net that contains the parameters of your model. More in detail, define the following:

-   A constructor that given `p` and `q` returns an object `shallow_net` with randomly initialized parameters `theta` and `beta`.

-   An S3 method `predict` that for a given object of the type `shallow_net` and a $n_2 \times p$ data matrix `X` returns a $n_2$ -vector with the predicted probabilities

-   An S3 method `train`, that for a given a $n \times p$ data matrix `X`, a $n$ -vector `y` of categorical outputs, the learning rate and number of iterations, uses gradient descent to learn the parameters of the neural
network.

#### 1.3 Re-run Examples 1 and 2 by using the redesigned code

## Problem 2

#### 2.1 Profile the S3 method `train`. Comment on the results.

#### 2.2 Re-implement in C++ the computation of the ***gradient of the parameter*** `theta` of the neural network. 

#### 2.3 Define a new S3 method `train_fast` that is a variation of `train` that replaces the code for the computation of the gradient of `theta` with the Cpp function implemented by calling 
`dL_dtheta <-compute_gradient_theta(X_aug, f_hat, y, beta, A)`

#### 2.4 Check that `train` and `train_fast` give the same results by re-running the Example 1 and 2 with
train_fast. Compare their performance.
