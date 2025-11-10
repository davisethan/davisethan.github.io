---
layout: default
title: Ethan Davis
---

# Introduction

![Ethan Davis](assets/images/ethan_davis.jpg)

**Email:** <a href="mailto:davise5@uw.edu">davise5@uw.edu</a>

I am pursuing a PhD in building scalable and reliable machine learning systems that draw on probabilistic models, distributed systems, and high performance computing. At the University of Washington I research uncertainty aware deep learning to strengthen AI deployed in safety critical applications. I have seven years of industry experience as a software engineer, building distributed systems, scalable cloud infrastructure, and resilient data pipelines.

<a href="assets/files/Ethan_Davis_CV_2025.pdf" target="_blank" rel="noopener noreferrer">Curriculum Vitae</a> | <a href="assets/files/Ethan_Davis_Resume_2025.pdf" target="_blank" rel="noopener noreferrer">Resume</a>

# Education

**M.S. Computer Science, University of Washington, 2024–2026**  
**B.S. Computer Science, Oregon State University, 2020–2022**  
**B.S. Mathematics, University of Portland, 2011–2015**

# Research Experience

## MS Thesis

The goal of my MS thesis is to evaluate when Bayesian ML benefits motor-imagery electroencephalogram (MI-EEG) brain-computer interface (BCI) classification [[3, 4]](#ms-thesis-references). Motivations for MI-EEG/BCI research include assisting neurorehabilitation therapies and controlling robotic prosthetics [[2]](#ms-thesis-references). I compare analogous frequentist and Bayesian ML models from the three most popular types of MI-EEG classifiers: firstly, linear spatial filters, or secondly, Riemannian geometric spatial filters, plus training models, and thirdly deep learning [[1]](#ms-thesis-references). Metrics used to measure predictive performance come from discrimination, calibration, and sharpness [[3]](#ms-thesis-references).

![End-to-End Pipeline](assets/images/e2e_pipeline.png)

My MS thesis experiment compares frequentist and Bayesian probability models. Frequentist models optimize parameters &#952; that minimize loss when predicting label y given example x. Bayesian models average over parameters &#952; making comprehensive predictions given all available information. Prior beliefs of the distribution for parameters &#952; and inference methods for approximating the often intractable evidence are additional requirements for Bayesian modeling [[3, 5]](#ms-thesis-references).

The datasets are popular options for MI-EEG modeling [[1]](#ms-thesis-references). The frequentist models are top-performing classifiers of these data in terms of accuracy [[1]](#ms-thesis-references). Our contribution considers the Bayesian analogs of these models and additional metrics. The acronyms of the models are as follows: common spatial pattern (CSP), tangent space (SP), linear discriminant analysis (LDA), support vector machine (SVM), logistic regression (LR), shallow convolutional neural network (SCNN), deep convolutional neural network (DCNN), Gaussian process (GP) [[2, 7, 3, 6, 5]](#ms-thesis-references).

The prior distributions range from light to heavy tailedness. Our choice of a Gaussian distribution is a standard baseline, though previous results have shown that heavy-tailed priors like the Laplace or Cauchy distributions are more expressive for training ML [[2]](#ms-thesis-references). Our experiment uses the three most common methods inference for ML: Laplace approximation, variational inference (VI), and sampling methods of which Hamiltonean Monte Carlo (HMC) is the golden standard [[3, 4]](#ms-thesis-references). These methods range from fastest and least accurate, to slowest and most accurate. Each technique has unique metrics that can be used to explain its performance, for example in terms of model improvement per computational unit: condition number of the Hessian matrix, evidence lower bound (ELBO), and effective sample size (ESS) for Laplace approximation, VI, and HMC respectively [[4, 8]](#ms-thesis-references).

![Experiment Design](assets/images/experiment_design.png)

The mother of all BCI benchmarks (MOABB) is a software library created for the purpose of reproducible BCI research [[1]](#ms-thesis-references). Its expert, minimal MI-EEG signal preprocessing, widely accepted methods of model selection, and rigorous statistical evaluation functions creates a standardized framework of BCI research experiment design allowing researchers to focus on designing novel ML models. The centralization of datasets offers model comparisons with statistically valid claims that some model/family empirically dominates another.

With my breadth of models, priors, and inferences, I can formally test multiple hypotheses for the purpose of evaluating when Bayesian ML benefits MI-EEG classification. My primary research question is whether Bayesian models outperform frequentist ones. Secondary research questions concern the effect of prior distribution tailedness, inference speed/accuracy tradeoff, and relative performance of the three types of MI-EEG classifiers.

### MS Thesis References

1. Chevallier, S., Carrara, I., Aristimunha, B., Guetschel, P., Sedlar, S., Lopes, B., Velut, S., Khazem, S., Moreau, T. (2021). The largest EEG-based BCI reproducibility study for open science: the MOABB benchmark. In J. Neural Eng.
2. Nam, C. S., Nijholt, A., & Lotte, F. (2018). Brain–Computer Interfaces Handbook: Technological and Theoretical Advances. CRC Press.
3. Bishop, C. M. (2016). Pattern recognition and machine learning. Springer.
4. Murphy, K. P. (2023). Probabilistic Machine Learning: Advanced Topics. MIT Press.
5. Deisenroth, M. P., Faisal, A. A., & Ong, C. S. (2020). Mathematics for machine learning. Cambridge University Press.
6. Bishop, C. M., & Bishop, H. (2023). Deep learning: Foundations and Concepts. Springer Nature.
7. Pennec, X., Sommer, S., & Fletcher, T. (2019). Riemannian Geometric statistics in medical image analysis. Academic Press.
8. Murphy, K. P. (2022). Probabilistic Machine Learning: An Introduction. MIT Press.

<!-- At the University of Washington, my research goal is to determine when Bayesian learning benefits real-time motor imagery electroencephalography (MI-EEG) brain–computer interfaces (BCIs). To support this, I am designing a framework that collects a suite of metrics and diagnostics for evaluating machine learning models. These data enable multiple hypotheses to be formally generated and tested. -->

# Technical Reports

|Paper|
|---|
|<a href="https://doi.org/10.48550/arXiv.2510.05027" target="_blank" rel="noopener noreferrer">Metaheuristic Algorithms for Combinatorial Optimization</a>|
|<a href="https://doi.org/10.48550/arXiv.2509.04594" target="_blank" rel="noopener noreferrer">High Performance Matrix Multiplication</a>|
|<a href="https://doi.org/10.5281/zenodo.17180358" target="_blank" rel="noopener noreferrer">Linear Algebra for Image Compression</a>|
|<a href="https://doi.org/10.5281/zenodo.17297422" target="_blank" rel="noopener noreferrer">Data Structures and Algorithms</a>|

# Software & Reproducibility

|Software|
|---|
|<a href="https://doi.org/10.5281/zenodo.17274214" target="_blank" rel="noopener noreferrer">Metaheuristic Algorithms for Combinatorial Optimization</a>|
|<a href="https://doi.org/10.5281/zenodo.17299758" target="_blank" rel="noopener noreferrer">High Performance Matrix Multiplication</a>|
|<a href="https://doi.org/10.5281/zenodo.17299529" target="_blank" rel="noopener noreferrer">Linear Algebra for Image Compression</a>|
|<a href="https://doi.org/10.5281/zenodo.17289626" target="_blank" rel="noopener noreferrer">Data Structures and Algorithms</a>|
|<a href="https://doi.org/10.5281/zenodo.17299086" target="_blank" rel="noopener noreferrer">Triangle Counting</a>|

# Teaching & Mentoring

Assisting Prof. Erika Parsons in updating the course _Mathematics for Machine Learning_, including curriculum design, textbook selection, and assignment development and grading. Independently of the course, I curated MI-EEG BCI research directions and reproducibility standards (MOABB, Riemannian pipelines, GNNs) to align lab efforts and accelerate studies.

# Selected Industry Experience

**Software Engineer**, SeekOut – Bellevue, WA (2022–2024)  
Built and optimized data pipelines and distributed systems in C# and Azure, applying object-oriented design, SOLID principles, and automated testing to improve scalability, reliability, and maintainability of large-scale ETL and search infrastructure.

**Software Engineer**, Independent Project – Seattle, WA (2017–2020)  
Designed a fault-tolerant, microservices-based web application using Java, Spring Boot, Node.js, and AWS, implementing distributed systems principles for scalable cloud deployment.

**Software Engineer**, StackBrew – Redmond, WA (2015–2017)  
Developed a JavaScript AST interpreter and backend microservices (Node.js, Go) deployed on GCP, exploring collaborative editing algorithms (OT and CRDTs) to inform distributed software design.

# Conference Posters

|Conference|
|---|
|<a href="assets/files/pumps_poster.pdf" target="_blank" rel="noopener noreferrer">PUMPS+AI 2025 ACM Europe Summer School</a>|

# Certifications

|Certification|
|---|
|<a href="assets/files/pumps_certification.pdf" target="_blank" rel="noopener noreferrer">PUMPS+AI 2025 Statement of Accomplishment in CUDA workshops</a>|
