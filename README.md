# CompTopoTrack

Official repository for the paper:

**Cell Tracking via Competition Topology-Aware Metric Learning and Geometry-Constrained Division Recovery**. Once the paper is accepted, we will make all training weights publicly available.

## Overview

CompTopoTrack is a cell-tracking framework for time-lapse microscopy. It is designed to address two recurring challenges in cell tracking: ambiguous associations among visually similar local candidates and recovery of division lineages after one-to-one continuation matching.

The framework contains three main components:

- **Competition-Topology-Aware Metric Learning** — aligns representation learning with the local candidate competition encountered by the downstream tracker.
- **Multi-Branch Spatiotemporal Association GNN** — combines topology-aware metric embeddings, object attributes, and relative motion for continuation association.
- **Geometry-Constrained Division Recovery** — recovers mother--daughter lineage relationships from newly initiated and terminated trajectories using scale- and geometry-aware constraints.

## Method at a Glance

The tracking pipeline follows four main stages:

1. **Candidate graph construction**  
   Direct candidate edges are constructed between adjacent frames using a spatial search range.

2. **Competition-topology-aware representation learning**  
   Competition components in the adjacent-frame candidate bipartite graph are used as a first-order approximation of the multi-frame reverse information funnel. Direct candidate relations, component closure, intra-frame mutual exclusion, and node influence are incorporated into a topology-weighted Multi-Similarity objective.

3. **GNN-based continuation association**  
   A multi-branch spatiotemporal GNN separately encodes appearance, object attributes, and relative motion, and predicts continuation probabilities for direct candidate edges. The predicted edges are filtered and assembled using one-to-one matching.

4. **Geometry-constrained division recovery**  
   Division relationships are recovered after continuation trajectories are assembled. Sister-cell candidates and candidate mother tracks are determined from cell scale and spatial relationships, with joint assignment used in ambiguous local components.

## Input

CompTopoTrack operates on **frame-wise target markers** together with the corresponding microscopy images.

The framework supports both:

- instance-mask targets; and
- point-target markers.

For point-target data, local image patches can be extracted around target centers for appearance encoding while the point markers are retained for tracking representation.

## Datasets

The experiments reported in the paper include:

- **Fluo-N2DH-SIM+**
- **DynamicNuclearNet**
- **Bacterial**
- **C2C12**
- **ISBI Particle Tracking Challenge (Vesicle)**

The datasets are distributed by their original providers and are **not redistributed by this repository**. Please obtain them from the corresponding official sources and follow their respective licenses and usage terms.

## Evaluation

The paper evaluates tracking performance using dataset-appropriate metrics, including:

- **TRA** and **AOGM** for graph-based cell tracking evaluation;
- false-positive and false-negative continuation/division errors;
- **Division F1** for division-recovery ablations;
- **AA** and **TE** for C2C12 point-target tracking; and
- **\(\alpha\)**, **\(\beta\)**, and **\(JSC_{\theta}\)** for the ISBI Vesicle analysis.

Please refer to the paper for the exact evaluation protocol and dataset splits.

## Reproducibility

This repository hosts the implementation and reproducibility materials associated with CompTopoTrack. Public release materials are maintained here and may be updated as the manuscript and codebase are finalized.

For reproducible experiments, please use the dataset-specific settings, preprocessing rules, and evaluation protocols corresponding to the paper. Dataset files and third-party resources remain subject to their original licenses.

## Repository Updates

The repository will continue to be maintained as the project progresses. Updates may include code refinements, documentation, configuration files, pretrained checkpoints, and evaluation utilities associated with the paper.

## Citation

If you find this work useful, please consider citing the paper.

Citation information will be updated after publication.

<!--
@article{zou_comptopotrack,
  title   = {Cell Tracking via Competition Topology-Aware Metric Learning and Geometry-Constrained Division Recovery},
  author  = {Zou, Xingyue and Wang, Xueyuan},
  journal = {To appear},
  year    = {2026}
}
-->

## Authors

**Xingyue Zou**  
School of Information Engineering  
Southwest University of Science and Technology  
Mianyang, Sichuan, China

**Xueyuan Wang**  
School of Information Engineering  
Southwest University of Science and Technology  
Mianyang, Sichuan, China

## Contact

For questions regarding this work, please contact:

**Xingyue Zou**  
Email: 1493540899@qq.com

## License

Please refer to the license file included in this repository for the terms governing the released source code and associated materials. Third-party datasets, software, and pretrained resources remain subject to their respective licenses.
