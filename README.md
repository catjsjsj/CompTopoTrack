# CompTopoTrack

Official repository for the paper:

**Cell Tracking via Competition Topology-Aware Metric Learning and Geometry-Constrained Division Recovery**

## Overview

CompTopoTrack is a cell tracking framework for microscopy image sequences, designed to address ambiguous associations between visually similar cells and the recovery of cell division lineages.

The framework consists of three main components:

- **Competition Topology-Aware Metric Learning**, which incorporates local candidate competition into representation learning to improve the discriminability of ambiguous cell instances.
- **Multi-Branch Spatiotemporal Association GNN**, which integrates metric embeddings, object attributes, and relative motion for ordinary cell association.
- **Geometry-Constrained Division Recovery**, which jointly considers sibling relationships and candidate mother tracks to recover cell division lineages.

## Code Availability

The source code, pretrained models, training configurations, and evaluation scripts will be made publicly available **after the acceptance of the paper**.

Please stay tuned for updates.

## Datasets

The experiments in the paper include:

- Fluo-N2DH-SIM+
- DynamicNuclearNet
- Bacterial
- C2C12
- ISBI Particle Tracking Challenge (Vesicle)

Detailed data preparation instructions will be provided together with the source code.

## Citation

Citation information will be added after the paper is accepted and published.

## Authors

**Xingyue Zou**  
School of Information and Control Engineering  
Southwest University of Science and Technology

**Xueyuan Wang**  
School of Information and Control Engineering  
Southwest University of Science and Technology

## Contact

For questions regarding this work, please contact:

Xingyue Zou  
Email: 1493540899@qq.com

---

The complete implementation will be released after paper acceptance.
