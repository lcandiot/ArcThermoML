# Supporting Julia Scripts for “Validating thermodynamic models of arc-magma differentiation and training neural networks for rapid thermodynamic property inference”

[![DOI](https://zenodo.org/badge/1073623371.svg)](https://doi.org/10.5281/zenodo.17380100)

This repository contains the **Julia scripts and configuration files** used in  
> Lorenzo G. Candioti, Chetan Nathwani, and Cyril Chelle-Michou (2025),  
> *Validating thermodynamic models of arc-magma differentiation and training neural networks for rapid thermodynamic property inference*

The scripts reproduce the analyses and figures presented in the article and are shared **for transparency and reproducibility**.  
They are *not* intended as a general-purpose software package or to be actively maintained.

---

## 📦 Contents

We provide the necessary pipelines in the
```
./scripts/
```
directory. All files with the prefix `Benchmark_` have been used to analyze the performance of network and MAGEMin calls. Training and cross-validation procedure was executed using the files with `ex_Train_Igneous` prefix. The other files in the script folder host functionality that is laoder the aforementioned main scripts.

Accompanying small data sets for the benchmarking, the laboratory experiment compilation as well as the MAGEMin predictions for the experimental liquids and conditions can be found in the respective
```
./data/
```
subdirectories.

---

## ⚙️ Requirements

- Julia **v1.11** or newer  
- Packages will be installed automatically from `Project.toml`  
- Operating system: Linux, macOS, or Windows

---

## ▶️ Quick start

### Clone the repository
Clone the repository and create your own user directory
```
git clone https://github.com/lcandiot/ArcThermoML.git
cd ArcThermoML
mkdir user
```

### Recreate the environment
Launch the julia REPL inside and recreate the environment for this project project
```
julia --project=./
```
```
using Pkg
Pkg.instantiate()
```

### Run example script
You can infer thermodynamic quantities of arc magmas by running
```
include("./scripts/ex_infer_props.jl")
```
As a result you should see the following ouptut printed in the Julia REPL
```
(X_liq = [0.5592579245567322; 0.1716310977935791; … ; 0.00857582502067089; 0.09265629202127457;;], ρ_sys = [2658.150390625], ϕ_liq = [0.4125232696533203], ϕ_flu = [0.0], ρ_liq = [2145.30615234375], ρ_flu = [0.0])
```

## 📊 Reproducing the main article figures

First download the required data from the Zenodo repository and store them somewhere (in the `./user` directory for example). The files to reproduce the figures are
```
./scripts/
    evaluate_crossValidation.jl
    evaluate_MAGEMin_Experiments.jl
    evaluate_Network_accuracy.jl
```
Simply adjust the path variable pointing to the location of the data and execute one of the scripts in the Julia REPL like
```
include("./scripts/evaluate_crossValidation.jl")
```
This will print the figure to screen and save a png in the `./user/figures` subdirectory.
