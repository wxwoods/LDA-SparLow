# Trace Quotient with Sparsity Priors for Learning Low Dimensional Image Representations

This work studies the problem of learning appropriate low dimensional image representations. We propose a generic
algorithmic framework, which leverages two classic representation learning paradigms, i.e., sparse representation and the trace
quotient criterion, to disentangle underlying factors of variation in high dimensional images. Specifically, we aim to learn simple
representations of low dimensional, discriminant factors by applying the trace quotient criterion to well-engineered sparse
representations. We construct a unified cost function, coined as the SPARse LOW dimensional representation (SparLow) function, for
jointly learning both a sparsifying dictionary and a dimensionality reduction transformation. The SparLow function is widely applicable
for developing various algorithms in three classic machine learning scenarios, namely, unsupervised, supervised, and semi-supervised
learning. In order to develop efficient joint learning algorithms for maximizing the SparLow function, we deploy a framework of sparse
coding with appropriate convex priors to ensure the sparse representations to be locally differentiable. Moreover, we develop an
efficient geometric conjugate gradient algorithm to maximize the SparLow function on its underlying Riemannian manifold.
Performance of the proposed SparLow algorithmic framework is investigated on several image processing tasks, such as 3D data
visualization, face/digit recognition, and object/scene categorization.

<font face="黑体">This code just is for LDA Sparlow.</font>

# Result
Impact of the regularizers to the recognition rate on the USPS digits:
![avatar](https://github.com/wxwoods/LDA-SparLow/blob/master/results/LDA-SparLow.png)

Trace of performance over optimization process initialized with different sparse coding methods on 15-Scenes dataset:
![avatar](https://github.com/wxwoods/LDA-SparLow/blob/master/results/LDA-SparLow2.png)

Averaged classification Rate (%) comparison on 15-Scenes dataset.The classifier is 1NN for the third column if not specified:
![avatar](https://github.com/wxwoods/LDA-SparLow/blob/master/results/LDA-SparLow3.png)


# Dataset

You can load dataset from [here](http://users.umiacs.umd.edu/~zhuolin/projectlcksvd.html).

# paper
You can load [paper](https://github.com/wxwoods/LDA-SparLow/blob/master/paper/Trace%20Quotient%20with%20Sparsity%20Priors%20for%20Learning%20Low%20Dimensional%20Image%20Representations.pdf) for detail.

