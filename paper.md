---
title: 'SGMCMCJax: a lightweight JAX library for stochastic gradient Markov chain Monte Carlo algorithms'
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
date: 14 January 2022
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

Markov Chain Monte Carlo (MCMC) algorithms are currently the gold standard for approximating integrals with Bayesian models. However, these algorithms become prohibitively expensive for modern-day large-scale datasets. Stochastic gradient MCMC (SGMCMC) is a popular approach to approximate these integrals in the large data setting. SGMCMC methods include of a wide range of algorithms that are regularly used to fit statistical and machine learning models, including Bayesian neural networks (BNNs). SGMCMCJax is a lightweight library of SGMCMC algorithms that can easily be used for both research purposes or practical applications.

# Statement of need

SGMCMCJax is a Python package written in the popular JAX library. Although there are libraries for SGMCMC algorithms in other languages and automatic differentiation frameworks (REF: sgmcmc-r, other?..), there is no SGMCMC library within the JAX ecosystem. However, as JAX has become a popular framework for machine learning and scientific computing, this gap has become more noticeable. As SGMCMC algorithms are a standard tool to train Bayesian neural networks, as well as other statistical models with large datasets, we have written this library of popular samplers to fill this gap.

SGMCMCJax uses JAX to perform automatic differentiation and compilation to XLA. The use of JAX allows the SGMCMCJax library to effortlessly run on GPUs and TPUs, which is essential for large models such as BNNs. As a result, the library utilizes an easy-to-use interface and provides very competitive speed performance. SGMCMCJax is desgined in a modular framework allowing users to simply run one of its many algorithms, or to create new algorithms for research purposes by using the exisiting algorithms as building blocks. Furthermore, SGMCMCJax can integrate easily with other codebases within the JAX ecosystem such as Flax, a neural network library for JAX.

# References

SGMCMCJax is written using JAX and relies on some aspects of the Python ecosystem such as numpy. ADD REFS

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
