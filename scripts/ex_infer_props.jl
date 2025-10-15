# Inferring arc magma properties with the neural network
using Lux, LuxCUDA, JLD2
include("MAGEMin_MLPs.jl")
using .MAGEMin_MLPs

function main_func()
    
    # Set conditions
    T       = 900.0       # T in [°C]
    P       = 5.0         # P in [kbar]
    oxides  = ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O+Na2O", "TiO2", "H2O"]
    X       = [ 52.1,   18.3,    10.5,  5.55,  7.24,  0.38+2.6,   0.68,   4.0]
    
    # Load surrogate model
    mpath  = "./data/surrogate_model/surrogate.jld2"
    model  = JLD2.load(mpath, "model")
    x_mean = JLD2.load(mpath, "x_mean")
    x_std  = JLD2.load(mpath, "x_std")
    y_mean = JLD2.load(mpath, "y_mean")
    y_std  = JLD2.load(mpath, "y_std")
    ig_NN  = (model=model, in_mean=x_mean, in_std=x_std, out_mean=y_mean, out_std=y_std)

    # Infer properties
    arc_props = MAGEMin_MLPs.infer_igneous_properties(ig_NN, T, P, X, oxides)

    # Return result
    return arc_props
end

# Run main
arc_props = main_func()