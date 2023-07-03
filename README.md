# PyHessian_modified
Some modifications based on Pyhessian: https://github.com/amirgholami/PyHessian

According to the power iteration method to compute the eigenvalues and eigenvactors of neural networks. I think there is something wrong with Pyhessian repository (details in Hessian_top_1.ipynb) and provide the method to compute the top-1 eigenvalue of each layer which is not in the Pyhessian.

To do list:
1. Compute the top-k eigenvalues and corresponding eigenvectors.
1. Compute the hessian trace of each layer. 
