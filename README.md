# iterative-ensemble-smoother
An iterative ensemble smoother (iES) based on regularized Levenburg-Marquardt, see the paper "Iterative Ensemble Smoother as an Approximate Solution to a Regularized Minimum-Average-Cost Problem: Theory and Applications ", by Luo et al., SPE-176023-PA, https://doi.org/10.2118/176023-PA

This depository contains an PYTHON implementation of the aforementioned iES, which is most of the time used in ensemble-based reservoir data assimilation (also known as history matching) problems. Our main purpose here is  to demonstrate how to use iES algorithm infer the input of a neural network(implemented with pytorch) . This code is based on [lanhill/Iterative-Ensemble-Smoother: An iterative ensemble smoother (iES) based on regularized Levenburg-Marquardt](https://github.com/lanhill/Iterative-Ensemble-Smoother) which is implemented with MATLAB. 

# Disclaimer

This depository is made available and contributed to under the license that include terms that, for the protection of the contributors, make clear that the depository is offered “as-is”, without warranty, and disclaiming liability for damages resulting from using the depository. This guide is no different. The open content license it is offered under includes such terms.

The code may include mistakes, and can’t address every situation. If there is any question, we encourage you to do your own research, discuss with your community or contact us. 

All views expressed here are from the contributors, and do not represent the opinions of any entities with which the contributors have been, are now or will be affiliated. 
