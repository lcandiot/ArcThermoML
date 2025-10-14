# Supporting Julia Scripts for “Validating thermodynamic models of arc-magma differentiation and training neural networks for rapid thermodynamic property inference”

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
```
git clone https://github.com/lcandiot/ArcThermoML.git
cd ArcThermoML
```

### Recreate the environment
```
julia --project -e "using Pkg; Pkg.instantiate()"
```

### Run example script
First, create a directory at the top level of this repository via
```
mkdir user
```
Next run 
