# Reproducing and Extending Revelio
Mherub Ahsan
Friedrich-Alexander University of Erlangen

This repository contains my PRL seminar reproduction and extension of selected Diff-C experiments from the paper:

**Revelio: Interpreting and Leveraging Semantic Information in Diffusion Models**

## Project goal

The project has two parts:

1. Reproduce the Stable Diffusion 1.5 Diff-C layer comparison on Oxford-IIIT Pet at timestep t=25.
2. Extend the experiment by testing missing intermediate timesteps t=50 and t=75 for the best reproduced layer, up_ft1.

## Method

Stable Diffusion 1.5 is used as a frozen feature extractor.  
The Diff-C classifier is trained on extracted diffusion features.

Pipeline:

Image → Stable Diffusion feature extractor → selected layer → GAP pooling → Diff-C classifier → top-1 test accuracy

## Experimental setup

| Setting | Value |
|---|---|
| Dataset | Oxford-IIIT Pet |
| Model | Stable Diffusion 1.5 |
| Classifier | Diff-C |
| Epochs | 30 |
| Batch size | 4 |
| Learning rate | 1e-4 |
| Prompt / pooling | empty / GAP |
| Metric | Top-1 test accuracy |

## Layer comparison at t=25

| Layer | Paper reference | My result | Difference |
|---|---:|---:|---:|
| bottleneck:0 | 69.97% | 71.736% | +1.77 pp |
| up_ft:0 | 73.29% | 74.244% | +0.95 pp |
| up_ft:1 | 88.61% | 87.244% | -1.37 pp |
| up_ft:2 | 81.63% | 79.204% | -2.43 pp |

## Timestep sweep for up_ft1

| Timestep | Accuracy |
|---:|---:|
| 25 | 87.244% |
| 50 | 86.863% |
| 75 | 86.890% |
| 100 | 86.999% |

## Main conclusion

The reproduction confirms the main qualitative trend of the paper: **up_ft1 is the strongest evaluated Stable Diffusion 1.5 layer** for this task.

The own experiment shows that the intermediate timesteps **t=50** and **t=75** are close to t=25, but they do not improve over t=25.

## HPC notes

The experiments were executed on the FAU/NHR TinyGPU HPC cluster using SLURM.

Some jobs failed on RTX 2080 Ti due to CUDA out-of-memory. The failed bottleneck and up_ft0 runs were rerun on V100 while keeping batch size 4 and the training settings fixed.

## Repository contents

- `jobs/`: SLURM job scripts
- `scripts/`: result collection and plotting scripts
- `results/`: CSV and summary result files
- `figures/`: generated plots
- `logs/final/`: selected final logs
- `evidence/`: screenshots used as execution evidence
- `report/`: final report files
- `presentation/`: final presentation files
