---
title: 'SGMCMCJAX: lightweight library of stochastic gradient Markov chain Monte Carlo algorithms written in JAX'
tags:
  - Python
  - JAX
  - MCMC
  - SGMCMC
  - Bayesian inference
authors:
  - name: Jeremie Coullon^[co-first author] # note this makes a footnote saying 'co-first author'
    orcid: 0000-0002-7032-3425
    affiliation: "1" # (Multiple affiliations must be quoted)
  - name: Christopher Nemeth
    orcid: 0000-0002-9084-3866
    affiliation: "2" # (Multiple affiliations must be quoted)
affiliations:
 - name: Cervest, London, UK
   index: 1
 - name: Lancaster University, UK
   index: 2
date: 14 February 2022
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Markov Chain Monte Carlo (MCMC) is currently the gold standard for approximating integrals needed in Bayesian inference. However as these algorithms becomes prohibitively expensive for large datasets, stochastic gradient MCMC (SGMCMC) is a popular approach to approximate these integrals in these cases. This approach consists of a wide range of algorithms that are regularly used to fit statistical models or Bayesian neural networks (BNNs). SGMCMCJax is a lightweight library of SGMCMC algorithms that can easily be used for both research purposes or practical applications.

# Statement of need

SGMCMCJax is a Python package written in the popular JAX library. Although there are libraries for SGMCMC algorithms in other languages and automatic differentiation frameworks (REF: sgmcmc-r, other?..), there is no mature library for the JAX ecosystem. However as this has become a popular framework for machine learning and scientific computing, this gap has become more noticeable. As SGMCMC algorithms are a standard tool to train Bayesian neural networks as well as statistical models with large datasets, we have written this library of samplers to fill this gap.

SGMCMCJax uses JAX to perform automatic differentiation and compilation to XLA. The use of JAX allows the library to easily run on GPUs and TPUs which is essential for large models such as BNNs. As a result, the library can have an easy to use interface while also having very competitive performance. SGMCMCJax is modular so can be used to simply run one of its many algorithm or to use as building blocks to build new algorithms for research purposes. Furthermore, SGMCMCJax can integrate easily with other codebases in the JAX ecosystem such as Flax, a neural network library for JAX.

SGMCMCJax was mainly designed for research purposes, as SGMCMC algorithms often include reusing building blocks to create new algorithms (such as gradient estimators, diffusions, or step size schedules).

# References

SGMCMCJax is written using JAX as relies on some aspects of the Python ecosystem such as numpy. ADD REFS

Although SGMCMCJax is a recent library it has already been used in a research paper (@coullon2021efficient) as well as used in the code to accompany the book "Machine learning: a probabilistic perspective" (github ref: https://github.com/probml/pyprobml).


# Acknowledgements

The design of the codebase was inspired inspired by JAX's optimizers module as well as the Blackjax library of MCMC samplers. We give special thanks to Kevin Murphy, Remi Louf, Colin Carol, Charles Matthews, and Sharad Vikram for code contributions and insightful discussions.

<!-- # Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)" -->



# References
