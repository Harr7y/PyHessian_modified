# PyHessian_modified
Some modifications based on Pyhessian: https://github.com/amirgholami/PyHessian

Compute the eigenvalues and eigenvactors of neural networks according to the power iteration method and provide the method to compute the top-1 eigenvalue and trace of each layer which is not in the Pyhessian.

1. Preliminary knowledge and Correctness proof are in Hessian_top_1.ipynb
2. A easy understanding version is function **eigen** in utils.py
3. demo.py is an illustration of how to compute the eigenvalue and trace with the given (input, label) pair.
4. You can change the function **get_params_grad** in utils.py to focus on the parameters which you are interested in.
