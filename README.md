# EXPOSE: Explainable and Domain-Robust Embeddings from Pathology Vision Foundation Models using Sparse Autoencoders

Accepted at [eXCV Workshop](https://excv-workshop.github.io/) of [ECCV 2026](https://eccv.ecva.net/).


## Abstract

Vision Foundation Models (VFMs) are widely used in computational pathology but remain sensitive to domain shifts arising from variations in staining, tissue preparation, and scanner hardware. A key limitation is that VFM embeddings entangle biological with domain-specific information, hindering cross-domain generalization. We propose Explainable Probing of Cross-Domain Sparse Embeddings (EXPOSE), a framework that uses Sparse Autoencoders (SAEs) as an explainable bottleneck to identify and suppress domain-specific components in VFM embeddings. We train a sparse representation of VFM features, use a linear classifier to identify domain-specific latent dimensions, and mask these features prior to downstream relapse prediction without retraining the backbone model. Experiments on a large prostate cancer dataset with multiple acquisition domains show that SAE features capture both domain- and task-specific information, which are partially disentangled in the latent space. Removing domain-specific features improves cross-domain performance and increases embedding robustness as measured by the Domain Robustness Index (DoRI).

---

Code is going to be published soon!

---
