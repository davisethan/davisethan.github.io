---
layout: default
title: Ethan Davis
---

![Ethan Davis](assets/images/profile.jpeg)

<a href="mailto:davise5@uw.edu">Contact</a> | <a href="assets/files/Ethan_Davis_CV_2025.pdf" target="_blank" rel="noopener noreferrer">Curriculum Vitae</a>

# Research Experience

## MS Thesis

Interested in probabilistic ML for trustworthy prediction, I focused my thesis on the effects of Bayesian models contrasted with frequentist baselines for motor imagery electroencephalogram (MI-EEG) classification in the SPANER Lab (Smart Platform for Augmented Neurology Rehabilitation) at the University of Washington [[1](#mst-1), [3](#mst-3), [4](#mst-4), [7](#mst-7)]. In a large-scale study comprising twenty datasets and six pairs of classifiers, I compared ML pipelines in terms of prediction performance using meta-analysis instruments to accumulate effect size estimates in proper scoring rules, discrimination, and calibration. Following my defense, I refined my experiment design and interpretation of results including the direction of future research [[2](#mst-2), [5](#mst-5), [6](#mst-6)]. I concluded that the effects of Bayesian complete-pooling on reliability/calibration and sharpness are statistically significant but not practically significant, with no detectable difference in overall performance, resolution, or discrimination. I hypothesize Bayesian partial-pooling models would generate larger effects.

### MS Thesis Artifacts

1. <span id="mst-1"></span>Davis, E. (2026). Benefits of being Bayesian: Motor imagery electroencephalogram classification (Publication No. 32735698) [Master's thesis, University of Washington]. ProQuest Dissertations & Theses Global.
2. <span id="mst-2"></span>Davis, E. (2026). Bayesian complete-pooling in cross-subject classification for motor imagery electroencephalogram (arXiv:2607.22980). arXiv. https://doi.org/10.48550/arXiv.2607.22980
3. <span id="mst-3"></span>Davis, E. (2026). Benefits of being Bayesian: Motor imagery EEG classification (Thesis software) (Version v1.0-thesis) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.20415596
4. <span id="mst-4"></span>Davis, E. (2026). Bayesian and frequentist motor imagery EEG classification: Model scores, compute profiling, and MCMC convergence diagnostics [Data set]. figshare. https://doi.org/10.6084/m9.figshare.32176734
5. <span id="mst-5"></span>Davis, E. (2026). Bayesian and frequentist pipeline benchmarking for cross-subject motor imagery EEG classification [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.21540429
6. <span id="mst-6"></span>Davis, E. (2026). Raw evaluation outputs for Bayesian complete-pooling in cross-subject classification for motor imagery electroencephalogram [Data set]. Zenodo. https://doi.org/10.5281/zenodo.21538705
7. <span id="mst-7"></span>Davis, E. (2026, June 1). Benefits of being Bayesian: Motor imagery electroencephalogram classification (MS thesis defense slides). Zenodo. https://doi.org/10.5281/zenodo.21724209

## Additional Research

Before committing to probabilistic machine learning research for my MS thesis I studied varying scopes of the machine learning systems stack, several beginning as course projects that I extended into independent work. Eigenfaces was a project in numerical linear algebra with convergence proofs for eigendecomposition and software in C [[1](#ar-1), [4](#ar-4)]. Chaining plain matrix multiplications introduced catastrophic cancellation through rounding of intermediate products, which I avoided by fusing the operations into single-pass loops specialized to each algorithm. For generalized matrix multiplication (GEMM) [[5](#ar-5), [6](#ar-6)], I studied NVIDIA CUDA and benchmarked duration between CPU/GPU and third-party/hand-rolled implementations across matrix sizes. Using distributed AWS EC2 clusters I researched and built ant colony optimization with Hadoop MapReduce [[2](#ar-2), [7](#ar-7)], and triangle counting with Apache Spark [[3](#ar-3)], both in Java. I also researched, designed, and built software architectures for end-to-end ML lifecycles with continuous training for ads recommendation systems in social networks.

### Additional Research Artifacts

1. <span id="ar-1"></span>Davis, E. (2025). Linear algebra for image compression. Zenodo. https://doi.org/10.5281/zenodo.17180358
2. <span id="ar-2"></span>Davis, E. (2025). Ant colony optimization with Hadoop MapReduce on distributed AWS EC2 clusters (Version v1.0.2) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.17274214
3. <span id="ar-3"></span>Davis, E. (2025). Triangle counting with Apache Spark on distributed AWS EC2 clusters (Version v1.0.1) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.17299086
4. <span id="ar-4"></span>Davis, E. (2025). Eigenfaces for image compression: Eigendecomposition in C (Version v1.0.1) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.17299529
5. <span id="ar-5"></span>Davis, E. (2025). High performance matrix multiplication: GEMM benchmarks across CPU and GPU (Version v1.0.1) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.17299758
6. <span id="ar-6"></span>Davis, E. (2025). High performance matrix multiplication (arXiv:2509.04594). arXiv. https://doi.org/10.48550/arXiv.2509.04594
7. <span id="ar-7"></span>Davis, E. (2025). Exploration-exploitation-evaluation (EEE): A framework for metaheuristic algorithms in combinatorial optimization (arXiv:2510.05027). arXiv. https://doi.org/10.48550/arXiv.2510.05027

# Teaching Experience

## Instruction and Materials

My advisor, Dr. Erika Parsons, invited me to design and deliver a guest lecture on feature engineering and selection for her graduate course, Data Mining for Machine Learning. The materials included a runnable Jupyter Notebook using motor imagery electroencephalogram (MI-EEG) data [[1](#im-1)], letting students execute a full classification pipeline themselves, alongside supplementary proofs of spatial filtering methods, including the common spatial pattern and tangent space mapping, grounded in linear algebra and Riemannian geometric statistics. As part of my thesis, I wrote detailed appendices released as lesson notes in Bayesian machine learning for MI-EEG [[2](#im-2)], spanning eleven sections, including EEG origins and handling protocols, Gaussian processes and Bayesian neural networks, Hamiltonian Monte Carlo, and meta-analysis for evidence synthesis, with derivations worked in full and figures reproduced throughout. I have also written a data structures and algorithms reference spanning two dozen problem categories, pairing worked solutions with complexity analyses [[3](#im-3), [4](#im-4)].

### Instruction and Materials Artifacts

1. <span id="im-1"></span>Davis, E. (2026, July 29). Feature engineering: Lesson materials. Zenodo. https://doi.org/10.5281/zenodo.21693577
2. <span id="im-2"></span>Davis, E. (2026, June 1). Bayesian machine learning for motor-imagery EEG: Lesson notes. Zenodo. https://doi.org/10.5281/zenodo.21711021
3. <span id="im-3"></span>Davis, E. (2026). Data structures and algorithms. Zenodo. https://doi.org/10.5281/zenodo.21693224
4. <span id="im-4"></span>Davis, E. (2025). Data structures and algorithms: Worked solutions and complexity analyses (Version v1.0.2) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.17289626

# Professional Service

## Research Software

The Mother of all BCI Benchmarks ([MOABB](https://moabb.neurotechx.com/docs/)) centralizes electroencephalogram (EEG) datasets and implements ML evaluations with an emphasis on reproducibility for brain-computer interface (BCI) experiments. I have discovered limitations that I documented as GitHub [issues](https://github.com/NeuroTechX/moabb/issues?q=is%3Aissue+author%3Adavisethan) and resolved in [pull requests](https://github.com/NeuroTechX/moabb/pulls?q=is%3Apr+author%3Adavisethan+is%3Amerged) merged upstream. Available in MOABB v1.5.0, these changes are credited in the [release notes](https://moabb.neurotechx.com/docs/whats_new.html), spanning fine-grained compute profiling of power, energy, water, and utilization across processors and memory, multi-metric scoring with arbitrary user-defined metrics, and fixes to model pickling and metric reporting. Robust hypothesis testing over these benchmarks draws on statistical methods most mature in R, so I have researched, built, and released two Python packages backed by a Docker Hub image for portability. The [tombolo](https://pypi.org/project/tombolo/) PyPI package exposes an R statistical runtime, containerized as the [tombolo](https://hub.docker.com/r/ethandavisecd/tombolo) image. The companion [moabbr](https://pypi.org/project/moabbr/) adapts MOABB to that interface. Both packages are pre-1.0 and under active development.

# Selected Industry Experience

## Software Engineer at SeekOut

<a href="https://www.seekout.com/" target="_blank" rel="noopener noreferrer">SeekOut</a> builds a search engine used by tech recruiters to find qualified candidates. When I worked there it was a unicorn startup of 150 employees, licensing data from LinkedIn, Google Scholar, and other applicant tracking systems (ATSs). On the data integration team I wrote extract, transform, load (ETL) pipelines carrying multiple sources into a centralized sink that staged records for search indexing, primarily using a C# .NET and Azure cloud computing stack. The recurring difficulty was reconciliation: every provider described the same candidate differently, with inconsistent fields and duplicate records, so normalizing heterogeneous schemas into one indexable representation drove most of the design. Alongside functional deliverables I led team efforts toward maintainable, SOLID services, using UML, documentation, and unit tests to drive decoupled, testable design that supported scaling the number of integrations and the throughput of data ingestion. That systems experience is what I now bring to building research infrastructure.

## Software Engineer at Independent Project

I built an image sharing platform end-to-end, owning both backend and frontend, and designing for production scale from the outset. The backend ran as Java Spring microservices, containerized with Docker and orchestrated by Kubernetes, alongside a React frontend with create, read, update, delete (CRUD) functionality served from Node.js web servers in the same cluster, with MongoDB outside it and AWS S3 for image storage. Reproducibility drove most of the infrastructure decisions. Provisioning infrastructure as a service (IaaS) on AWS with HashiCorp Terraform and managing configuration with Ansible meant the entire environment could be rebuilt from scratch rather than maintained by hand, eliminating configuration drift. That discipline carries directly into research, where a result is only as trustworthy as the environment that produced it. The platform was not sustainable to run on its own, but the infrastructure work has served me since.

## Software Engineer at StackBrew

<a href="https://www.stackbrew.com/" target="_blank" rel="noopener noreferrer">StackBrew</a> was a micro-startup building a platform for automated development, staging, and production environments of software engineering in the cloud. With only three to five engineers, I owned three microservices, all aimed at making a browser-based editor behave like a local development environment. The first was an abstract syntax tree (AST) interpreter for JavaScript that displayed variable contents as tooltips on hover for development and debugging, written in Node.js and later rewritten in C++ as a Node.js addon once interpretation became the bottleneck. The second handled version control in the editor and kicked off the automated CI/CD pipeline. The third supported collaborative editing using conflict-free replicated data types (CRDTs), which reconcile concurrent edits without a central arbiter or lock. I worked primarily in Node.js and Go with MongoDB, owning each service from design through deployment on Docker, Kubernetes, and GCP.
