# Supporting Julia Scripts for “Validating thermodynamic models of arc-magma differentiation and training neural networks for rapid thermodynamic property inference”

This repository contains the **Julia scripts and configuration files** used in  
> Lorenzo G. Candioti, Chetan Nathwani, and Cyril Chelle-Michou (2025),  
> *Validating thermodynamic models of arc-magma differentiation and training neural networks for rapid thermodynamic property inference*

The scripts reproduce the analyses and figures presented in the article and are shared **for transparency and reproducibility**.  
They are *not* intended as a general-purpose software package or to be actively maintained.

---

## 📦 Contents

---

## ⚙️ Requirements

- Julia **v1.10** or newer  
- Packages will be installed automatically from `Project.toml`  
- Operating system: Linux, macOS, or Windows

---

## ▶️ Quick start

### Clone the repository
```
git clone https://github.com/<yourname>/ArcThermoML.git
cd arc-thermo-ml-scripts
```

### Recreate the environment
```
julia --project -e "using Pkg; Pkg.instantiate()"
```

### Run example script (reproduces a key figure)
```
julia --project scripts/reproduce_Fig3.jl
```
