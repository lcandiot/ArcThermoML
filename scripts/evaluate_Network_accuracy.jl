# Original file name in my system: MAGEMin_MLPs/src/ex_evaluate_results_v6.jl
# To run this script and to create the figures, you need to download the data from Zenodo
using JLD2
using Lux, Statistics, Random
using MAGEMin_C
using CairoMakie, Base, Printf
include("MAGEMin_MLPs.jl")
using .MAGEMin_MLPs

set_theme!(theme_latexfonts())

# Tolerances
const ϵ_log = 1e-6
const ϵ_liq = 1e-3
const ϵ_flu = 1e-3
const DatType = Float64

@views function evaluate_results()

    # Randomness
    seed = 123
    rng  = Random.MersenneTwister()
    Random.seed!(rng, seed)

    # Set data path
    data_dir = "./user/zenodo_data" # <-- Path to Zenodo data
    mname = "TrainedModel_256HD_3HL_16384batchsize_1000epochs_full.jld2"

    # Switches
    draw_fig1 = true
    draw_fig2 = true
    log_transform = true

    # Load data and extract info
    info = JLD2.load("$(data_dir)/$mname", "/Info")

    # Unpack the model and test data
    model = JLD2.load("$data_dir/$mname", "MLP/model")
    x_mean_train = JLD2.load("$(data_dir)/$(mname)", "/Scaling/mean_input")
    x_std_train = JLD2.load("$(data_dir)/$(mname)", "/Scaling/std_input")
    y_mean_train = JLD2.load("$(data_dir)/$(mname)", "/Scaling/mean_output")
    y_std_train = JLD2.load("$(data_dir)/$(mname)", "/Scaling/std_output")

    # Make a prediction for data the network really has never seen before
    dataBase      = "ig"
    Xoxides       = ["SiO2"; "Al2O3"; "CaO"; "MgO"; "FeO"; "O"; "K2O"; "Na2O"; "TiO2"; "Cr2O3"; "H2O"]
    RC158c        = [ 52.1,   18.3, 10.5, 5.55,  7.24,  0.0,  0.38,   2.6,   0.68,    0.0,   4.0]
    RC158c      ./= sum(RC158c)
    sys_in        = "wt"
    data          = Initialize_MAGEMin(dataBase, verbose = -1);
    ph_rm   = remove_phases(["fl"], "ig")
    Out_PT        = single_point_minimization(5.0, 950.0, data, X = RC158c, Xoxides = Xoxides, sys_in = sys_in, rm_list = ph_rm)
    display(Out_PT.oxides)
    Finalize_MAGEMin(data)
    in_test = Matrix{DatType}(undef, 10, 1)
    in_test .= DatType.([950.0, 5.0, RC158c[1], RC158c[2], RC158c[3], RC158c[4], RC158c[5], RC158c[7]+RC158c[8], RC158c[9], RC158c[11]])
    if log_transform
        in_test[[2, 5, 6, 8, 9], :] .= log.(in_test[[2, 5, 6, 8, 9], :] .+ DatType(ϵ_log))
    end
    in_test_sc = MAGEMin_MLPs.TransformData(in_test, transform = :zscore, dims = 2, scale = true, mean_data = x_mean_train, std_data = x_std_train; DatType = DatType)
    states_t = Lux.testmode(model.st)
    (p̂1, p̂2, ŷ1, ŷ2, ŷ3, ŷ4, ŷ5, ŷ6) = Lux.apply(model.model.u, DatType.(in_test_sc), model.ps, states_t)[1]
    ŷ_sc = vcat(p̂1, vcat(p̂2, vcat(ŷ1, vcat(ŷ2, vcat(ŷ3, vcat(ŷ4, vcat(ŷ5, ŷ6)))))))
    ŷ  = MAGEMin_MLPs.TransformDataBack(ŷ_sc; transform = :zscore, dims=2, mean_data = y_mean_train, std_data = y_std_train)
    if log_transform
        ŷ[[5, 7, 8, 9], :] .= exp.(ŷ[[5, 7, 8, 9], :]) .- DatType(ϵ_log)
    end
    ŷ[3:10]     ./= sum(ŷ[3:10])
    y = zeros(DatType, size(ŷ))
    y[3, 1] = Out_PT.bulk_M_wt[1]
    y[4, 1] = Out_PT.bulk_M_wt[2]
    y[5, 1] = Out_PT.bulk_M_wt[3]
    y[6, 1] = Out_PT.bulk_M_wt[6] + Out_PT.bulk_M_wt[7]
    y[7, 1] = Out_PT.bulk_M_wt[4]
    y[8, 1] = Out_PT.bulk_M_wt[5]
    y[9, 1] = Out_PT.bulk_M_wt[8]
    y[10, 1] = Out_PT.bulk_M_wt[11]
    y[11,1] = Out_PT.rho
    y[12,1] = Out_PT.frac_M_wt
    y[13,1] = Out_PT.frac_F_wt
    y[14,1] = Out_PT.rho_M
    y[15,1] = Out_PT.rho_F
    liq_tot = sum(y[3:10])
    y[3:10] ./= liq_tot
    display(y)
    display(ŷ)
    display(abs.(y .- ŷ) ./ max.(y, ŷ) .* 100.0)

    # Load test data input and output data
    x_test_orig, y_test_orig = MAGEMin_MLPs.LoadData(data_path = "$data_dir/SynGEOROC_testing_data_5iniLev_2sub_lev_45.0_71.0wtSi_652.0_1298.0C_0.12_9.98kbar.jld2")

    y_test_ord = deepcopy(y_test_orig)

    # Reorder stuff - same reordering as training data (see ex_Train_Igneous_crossval_paper_v5.jl line 256)
    y_test_ord[[5, 6, 7, 8, 11, 12, 13, 14],    :] .= y_test_orig[[5, 8, 6, 7, 13, 11, 12, 14],   :]

    # Randomly pic a couple of test points
    Npts = size(y_test_ord, 2)
    rperm = randperm(rng, Npts)
    idxs = rperm[1:5:end]
    x_test = x_test_orig[:, idxs]
    y_test = y_test_ord[:, idxs]

    println("No. of randomly selected points : $(length(idxs))")

    # Sort for increasing melt fraction
    idx_liq_sort = sortperm(y_test[12, :], rev = true)
    y_test .= y_test[:, idx_liq_sort]
    x_test .= x_test[:, idx_liq_sort]

    # Log transform
    if log_transform
        x_test[[2, 5, 6, 8, 9], :] .= log.(x_test[[2, 5, 6, 8, 9], :] .+ DatType(ϵ_log))
    end

    # Prepare data and make a prediction
    x_test_sc = MAGEMin_MLPs.TransformData(x_test, transform = :zscore, dims = 2, scale = true, mean_data = x_mean_train, std_data = x_std_train; DatType = DatType)
    states_t = Lux.testmode(model.st)
    tic = Base.time()
    (p̂1, p̂2, ŷ1, ŷ2, ŷ3, ŷ4, ŷ5, ŷ6) = Lux.apply(model.model.u, DatType.(x_test_sc), model.ps, states_t)[1]
    @show toc = Base.time() - tic
    ŷ_sc = vcat(p̂1, vcat(p̂2, vcat(ŷ1, vcat(ŷ2, vcat(ŷ3, vcat(ŷ4, vcat(ŷ5, ŷ6)))))))
    ŷ  = MAGEMin_MLPs.TransformDataBack(ŷ_sc; transform = :zscore, dims=2, mean_data = y_mean_train, std_data = y_std_train)

    # Postprocess inferences ---------------------------------------------------------------------
    if log_transform
        x_test[[2, 5, 6, 8, 9], :] .= exp.(x_test[[2, 5, 6, 8, 9], :]) .- DatType(ϵ_log)
        ŷ[[5, 7, 8, 9],       :] .= exp.(ŷ[[5, 7, 8, 9],       :]) .- DatType(ϵ_log)
    end
    # Renormalize composition
    ŷ[3:10,:]     ./= sum(ŷ[3:10,:], dims = 1)

    ŷ .= DatType.(ŷ)

    # Generate fluid mask
    liquid_mask = ŷ[1, :] .>= 0.5
    fluid_mask  = ŷ[2, :] .>= 0.5
    for i in 3:10
        ŷ[i, :] .*= liquid_mask
    end
    ŷ[12, :] .*= liquid_mask
    ŷ[13, :] .*= fluid_mask
    ŷ[14, :] .*= liquid_mask
    ŷ[15, :] .*= fluid_mask

    # Correct points that are close to the aggregate transitions
    thres = 5e-2
    ph_mob      = vec(sum(ŷ[[12, 13], :], dims = 1))
    ph_mob_true = vec(sum(y_test[[12, 13], :], dims = 1))
    idx_solid  = findall(x -> x                    < 1e-3,       ph_mob)
    idx_molten = findall(x -> x                    > 1.0 - thres, ph_mob)
    idx_solid_true  = findall(x -> x                    < 1e-3,       ph_mob_true)
    idx_molten_true = findall(x -> x                    > 1.0 - thres, ph_mob_true)
    ŷ[[12, 13, 14, 15], idx_solid       ] .= 0.0
    y_test[[12, 13, 14, 15], idx_solid_true       ] .= 0.0
    idx_dens_melt = ŷ[14, :] .> ŷ[11, :]
    ŷ[14, idx_dens_melt] .= ŷ[11, idx_dens_melt]
    idx_dens_melt = y_test[14, :] .> y_test[11, :]
    y_test[14, idx_dens_melt] .= y_test[11, idx_dens_melt]
    ρ_sol_pred = (ŷ[11, :] .- ŷ[12, :] .* ŷ[14, :] .- ŷ[13, :] .* ŷ[15, :]) ./ (1.0 .- ŷ[12, :] .- ŷ[13, :])
    ρ_sol_pred[idx_molten] .= 0.0

    ρ_sol_true = (y_test[11, :] .- y_test[12, :] .* y_test[14, :] .- y_test[13, :] .* y_test[15, :]) ./ (1.0 .- y_test[12, :] .- y_test[13, :])
    ρ_sol_true[idx_molten_true] .= 0.0

    idx = findall(x -> x < 0.0, ρ_sol_true)
    display(y_test[:, idx])
    display(x_test[:, idx])
    display(minimum(ρ_sol_true))
    display(maximum(ρ_sol_true))
    display(minimum(ρ_sol_pred))
    display(maximum(ρ_sol_pred))

    # Calculate viscosity from Giordano et al. 2008 assuming a Na2O/K2O split factor
    visc_N = Vector{Float64}(undef, size(ŷ, 2))
    visc_M = Vector{Float64}(undef, size(ŷ, 2))
    for idx in axes(ŷ, 2)
        wt_vec_N = [ŷ[i, idx] for i in 3:10]
        wt_vec_M = [y_test[i, idx] for i in 3:10]
        T = x_test[1, idx]
        out_η_N = viscosity_GRD08_reduced(wt_vec_N; temp_C=T, alpha_Na=0.5, do_plot=false)
        out_η_M = viscosity_GRD08_reduced(wt_vec_M; temp_C=T, alpha_Na=0.5, do_plot=false)
        visc_N[idx] = out_η_N.viscosity_Pa_s
        visc_M[idx] = out_η_M.viscosity_Pa_s
    end

    # Testing model # [SiO2, Al2O3, CaO, MgO, FeO, (Na2O+K2O), TiO2, H2O]
    # wt_test = [52.7, 17.62, 3.23, 1.41, 9.4, 5.9+6.74, 1.64, 0.1]
    # α_test = 5.9/(5.9+6.74)
    # "TiO2": 1.64,
    # "Al2O3": 17.62,
    # "FeO": 9.4,
    # "Fe203": 0.0,
    # "MnO": 0.0,
    # "MgO": 1.41,
    # "CaO": 3.23,
    # "Na2O": 5.9,
    # "K2O": 6.74,
    # "P2O5": 0.0,
    # "H2O": 0.1,
    # "F2O_1": 0.00
    # out_test = viscosity_GRD08_reduced(wt_test; temp_C=1100.0, alpha_Na=α_test, do_plot=false)
    # @printf("Viscosity test : Giordano orig = %.4e \t My implementation: %.4e\n", 5452.070, out_test.viscosity_Pa_s)
    # f = Figure()
    # ax = Axis(f[1,1][1,1], xscale = log10, yscale = log10, xlabel =L"$\eta$ MAGE", ylabel = L"$\eta$ MLP")
    # sc1 = scatter!(ax, visc_M, visc_N, color = vec(y_test[12, :]), colormap = Reverse(:bilbao), alpha = 1.0)
    # Colorbar(f[1,1][1,2], colormap = Reverse(:bilbao), label = L"$\Phi$ MAGE")
    # display(f)
    # f = Figure(size = (500, 500), figure_padding = 20)
    # ax = Axis(f[1, 1], xlabel = "ρ_MAGE", ylabel = "ρ_MLP")
    # sc = scatter!(ax, ρ_sol_true, ρ_sol_pred)
    # display(f)
    # --------------------------------------------------------------------------------------------

    # Open figure 1
    ncol = 4
    nrow = 4
    labels = ["Class. (liq)", "Class. (flu)", "SiO2", "Al2O3 (liq)", "CaO (liq)", "MgO (liq)", "FeO (liq)", "K2O+Na2O (liq)", "TiO2 (liq)", "H2O (liq)", "Φ (liq)", "ρ (tot)", "ρ (liq)", "Φ (flu)", "ρ (flu)", "ρ (sol)"]
    fg1 = Figure(size = (1000, 1000))

    axs = []
    for col in 1:ncol
        for row in 1:nrow
            if col == 1 && row < nrow
                ax = Axis(fg1[row, col][1,1], ylabel = L"$$MLP", aspect = 1.0)
                push!(axs, ax)
            elseif col == 1 && row == nrow
                ax = Axis(fg1[row, col][1,1], xlabel = L"$$MAGEMin", ylabel = L"$$MLP", aspect = 1.0)
                push!(axs, ax)
            elseif row == nrow && col > 1
                ax = Axis(fg1[row, col][1,1], xlabel = L"$$MAGEMin", aspect = 1.0)
                push!(axs, ax)
            elseif row == nrow - 1 && col == ncol
                ax = Axis(fg1[row, col][1,1], xlabel = L"$$MAGEMin", aspect = 1.0)
                push!(axs, ax)                
            else
                ax = Axis(fg1[row, col][1,1], aspect = 1.0)
                push!(axs, ax)
            end
        end
    end
    # Fill figure prediction figure
    if draw_fig1
        cmap = :acton
        α    = 0.1
        scs = []
        for (idx_Ax, Ax) in enumerate(axs)
            idx_Ax > length(labels) -1 ? break : nothing
            if idx_Ax < 14
                sc = scatter!(Ax, y_test[idx_Ax + 2, :], ŷ[idx_Ax + 2, :], colormap = cmap, color = :gray, alpha = α)
                push!(scs, sc)
            end
            ln = lines!(Ax, [1e-5, 10000], [1e-5, 10000], linewidth = 3, color = :goldenrod4)
        end
        sc = scatter!(axs[14], ρ_sol_true, ρ_sol_pred, colormap = cmap, color = vec(y_test[12, :]), alpha = 1.0)
        push!(scs, sc)
        ln = lines!(axs[14], [1e-5, 10000], [1e-5, 10000], linewidth = 3, color = :goldenrod4)
        sc = scatter!(axs[15], visc_M, visc_N, color = vec(y_test[12, :]), colormap = cmap, alpha = 1.0)
        ln = lines!(axs[15], [1.0, 1e20], [1.0, 1e20], linewidth = 3, color = :goldenrod4)
        push!(scs, sc)
        # Create nice labels
        lsize_leg = 10
        spl_labels = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t"]
        axs[1].limits      = (0.4, 0.8, 0.4, 0.8)
        axs[2].limits      = (0.08, 0.25, 0.08, 0.25)
        axs[3].limits      = (-0.001, 0.13, -0.001, 0.13)
        axs[3].xticks      = 0:0.03:0.15
        axs[3].yticks      = 0:0.03:0.15
        axs[4].limits      = (0.04, 0.16, 0.04, 0.16)
        axs[5].limits      = (-0.001, 0.13, -0.001, 0.13)
        axs[6].limits      = (-0.001, 0.13, -0.001, 0.13)
        axs[7].limits      = (-0.001, 0.03, -0.001, 0.03)
        axs[7].xticks      = 0:0.01:0.08
        axs[7].yticks      = 0:0.01:0.08
        axs[8].limits      = (-0.001, 0.16, -0.001, 0.16)
        axs[9].limits      = (2000, 3200, 2000, 3200)
        axs[9].xticks     = 2000:300:3600
        axs[9].yticks     = 2000:300:3600
        axs[10].limits      = (-0.001, 1.001, -0.001, 1.001)
        axs[11].limits     = (0.0, 0.06, 0.0, 0.06)
        axs[11].xticks     = 0:0.02:0.09
        axs[11].yticks     = 0:0.02:0.09
        axs[12].limits     = (2000, 2600, 2000, 2600)
        axs[12].xticks     = 2000:200:3600
        axs[12].yticks     = 2000:200:3600
        axs[13].limits     = (-0.001, 1000, -0.001, 1000)
        axs[13].xticks     = 0:200:1000
        axs[13].yticks     = 0:200:1000
        axs[14].limits     = (2500.0, 3500.0, 2500.0, 3500.0)
        axs[14].xticks     = 2500:500:3500
        axs[14].yticks     = 2500:500:3500
        axs[15].limits     = (1e2, 1e16, 1e2, 1e16)
        axs[15].xscale     = log10
        axs[15].yscale     = log10
        axs[15].xticks = ([10^2, 10^6, 10^10, 10^14],[L"$10^{2}$", L"$10^{6}$", L"$10^{10}$", L"$10^{14}$"])
        axs[15].yticks = ([10^2, 10^6, 10^10, 10^14],[L"$10^{2}$", L"$10^{6}$", L"$10^{10}$", L"$10^{14}$"])
        scs[1].label  = L"SiO$_2$ [wt]"
        scs[2].label  = L"Al$_2$O$_3$ [wt]"
        scs[3].label  = L"$$CaO [wt]"
        scs[4].label  = L"$$MgO [wt]"
        scs[5].label  = L"$$FeO [wt]"
        scs[6].label  = L"K$_2$O+Na$_2$O [wt]"
        scs[7].label  = L"TiO$_2$ [wt]"
        scs[8].label  = L"H$_2$O [wt]"
        scs[9].label = L"$\rho$ Sys [kg.m$^{-3}$]"
        scs[10].label = L"$\phi$ Liq [ ]"
        scs[11].label = L"$\phi$ Flu [ ]"
        scs[12].label = L"$\rho$ Liq [kg.m$^{-3}$]"
        scs[13].label = L"$\rho$ Flu [kg.m$^{-3}$]"
        scs[14].label = L"$\rho$ Sol [kg.m$^{-3}$]"
        scs[15].label = L"$\eta$ Liq [Pa.s]"
        axislegend(axs[1],  position = (0.8, 0.05), labelsize = lsize_leg, halign = :right, backgroundcolor = (:white, 0.0), framecolor = (:black, 0.5), framevisible = false)
        axislegend(axs[2],  position = (0.8, 0.05), labelsize = lsize_leg, halign = :right, backgroundcolor = (:white, 0.0), framecolor = (:black, 0.5), framevisible = false)
        axislegend(axs[3],  position = (0.8, 0.05), labelsize = lsize_leg, halign = :right, backgroundcolor = (:white, 0.0), framecolor = (:black, 0.5), framevisible = false)
        axislegend(axs[4],  position = (0.8, 0.05), labelsize = lsize_leg, halign = :right, backgroundcolor = (:white, 0.0), framecolor = (:black, 0.5), framevisible = false)
        axislegend(axs[5],  position = (0.8, 0.05), labelsize = lsize_leg, halign = :right, backgroundcolor = (:white, 0.0), framecolor = (:black, 0.5), framevisible = false)
        axislegend(axs[6],  position = (0.8, 0.05), labelsize = lsize_leg, halign = :right, backgroundcolor = (:white, 0.0), framecolor = (:black, 0.5), framevisible = false)
        axislegend(axs[7],  position = (0.8, 0.05), labelsize = lsize_leg, halign = :right, backgroundcolor = (:white, 0.0), framecolor = (:black, 0.5), framevisible = false)
        axislegend(axs[8], position = (0.8, 0.05), labelsize = lsize_leg, halign = :right, backgroundcolor = (:white, 0.0), framecolor = (:black, 0.5), framevisible = false)
        axislegend(axs[9], position = (0.8, 0.05), labelsize = lsize_leg, halign = :right, backgroundcolor = (:white, 0.0), framecolor = (:black, 0.5), framevisible = false)
        axislegend(axs[10], position = (0.8, 0.05), labelsize = lsize_leg, halign = :right, backgroundcolor = (:white, 0.0), framecolor = (:black, 0.5), framevisible = false)
        axislegend(axs[11], position = (0.8, 0.05), labelsize = lsize_leg, halign = :right, backgroundcolor = (:white, 0.0), framecolor = (:black, 0.5), framevisible = false)
        axislegend(axs[12], position = (0.8, 0.05), labelsize = lsize_leg, halign = :right, backgroundcolor = (:white, 0.0), framecolor = (:black, 0.5), framevisible = false)
        axislegend(axs[13], position = (0.8, 0.05), labelsize = lsize_leg, halign = :right, backgroundcolor = (:white, 0.0), framecolor = (:black, 0.5), framevisible = false)
        axislegend(axs[14], position = (0.8, 0.05), labelsize = lsize_leg, halign = :right, backgroundcolor = (:white, 0.0), framecolor = (:black, 0.5), framevisible = true)
        axislegend(axs[15], position = (0.8, 0.05), labelsize = lsize_leg, halign = :right, backgroundcolor = (:white, 0.0), framecolor = (:black, 0.5), framevisible = true)
        for (idx, Ax) in enumerate(axs)
            if idx < 16
                text!(Ax, spl_labels[idx], position = (0.05, 0.9), space = :relative)
            end
        end
        # Add colorbar
        GL = fg1[4, 4] = GridLayout()
        Colorbar(GL[1,1][1,1], colormap = cmap, limits = extrema(vec(y_test[12, :])), vertical = false, flipaxis = false, label = L"$\phi$ Liq", tellheight = false, height = 16)
        hidedecorations!(axs[16])
        hidespines!(axs[16])
        display(fg1)

        # Save the figure
        if !isdir("./user/figures")
            mkpath("./user/figures")
        end
        save("./user/figures/Figure_9_MLP_Test.png", fg1, px_per_unit = 4)
    end

    # Error estimates
    if draw_fig2
        # Classification from probabilities
        τ = 0.5
        liquid_true = y_test[1, :] .>= τ
        fluid_true  = y_test[2, :] .>= τ
        liquid_pred = ŷ[1, :] .>= τ
        fluid_pred  = ŷ[2, :] .>= τ

        # Confusion matrices
        TP_liq = count(liquid_true .& liquid_pred)
        FP_liq = count(.!liquid_true .& liquid_pred)
        FN_liq = count(liquid_true .& .!liquid_pred)
        TN_liq = count(.!liquid_true .& .!liquid_pred)
        TP_flu = count(fluid_true .& fluid_pred)
        FP_flu = count(.!fluid_true .& fluid_pred)
        FN_flu = count(fluid_true .& .!fluid_pred)
        TN_flu = count(.!fluid_true .& .!fluid_pred)
        cm_liq = [TP_liq FP_liq; FN_liq TN_liq] ./ size(y_test, 2) .* 100.0
        cm_flu = [TP_flu FP_flu; FN_flu TN_flu] ./ size(y_test, 2) .* 100.0
        
        # Correct prediction mask
        correct_mask = (liquid_true .== liquid_pred) .& (fluid_true .== fluid_pred)
        Err = y_test[3:15, correct_mask] .- ŷ[3:15, correct_mask]
        MAE = mean(abs.(Err), dims = 2)
        RMSE = sqrt.(mean((y_test[3:15, correct_mask] .- ŷ[3:15, correct_mask]).^2, dims = 2))
        println("MAE")
        display(MAE)
        println("RMSE")
        display(RMSE)
        σAE = std(abs.(Err), dims = 2)
        σAE_pos = [std(row[row .> m]) for (row, m) in zip(eachrow(Err), MAE)]
        σAE_neg = [std(row[row .< m]) for (row, m) in zip(eachrow(Err), MAE)]

        # Open figure 2
        fg2 = Figure(size = (500, 500))
        ax1 = Axis(fg2[1,1], aspect = 1.681, ylabel = L"$$Error [wt-%]", tellheight = false, tellwidth = false)
        ax2 = Axis(fg2[1,2], aspect = 1.681, ylabel = L"$$Error [wt-%]", tellheight = false, tellwidth = false)
        ax3 = Axis(fg2[2,1], aspect = 1.681, ylabel = L"$$Error [wt-%]", tellheight = false, tellwidth = false)
        ax4 = Axis(fg2[2,2], aspect = 1.681, ylabel = L"Error [kg.m$^{-3}$]", tellheight = false, tellwidth = false)
        GL = fg2[3, 1:2] = GridLayout()
        ax5 = Axis(GL[1,1], aspect = 1.0, tellheight = true, tellwidth = true)
        ax6 = Axis(GL[1,2], aspect = 1.0, tellheight = true, tellwidth = true)
        coords = [(1,1), (2,1), (1,2), (2,2)]
        cm_lab = ["TP", "FP", "FN", "TN"]
        sc1 = scatter!(ax1, MAE[1:3] .* 100.0)
        err = errorbars!(ax1, [1, 2, 3], MAE[1:3] .* 100.0, σAE_neg[1:3] .* 100.0, σAE_pos[1:3] .* 100.0; whiskerwidth = 8)
        sc2 = scatter!(ax2, MAE[4:8].* 100.0)
        err = errorbars!(ax2, [1,2,3,4,5], MAE[4:8] .* 100.0, σAE_neg[4:8] .* 100.0, σAE_pos[4:8] .* 100.0; whiskerwidth = 8)
        sc4 = scatter!(ax3, MAE[[10, 11]] .* 100.0)
        err = errorbars!(ax3, [1,2], MAE[[10, 11]] .* 100.0, σAE_neg[[10, 11]] .* 100.0, σAE_pos[[10, 11]] .* 100.0; whiskerwidth = 8)
        sc4 = scatter!(ax4, MAE[[9, 12, 13]])
        err = errorbars!(ax4, [1,2,3], MAE[[9, 12, 13]], σAE_neg[[9, 12, 13]], σAE_pos[[9, 12, 13]]; whiskerwidth = 8)
        hm1 = heatmap!(ax5, round.(cm_liq)', colormap = CairoMakie.Reverse(:berlin))
        hm2 = heatmap!(ax6, round.(cm_flu)', colormap = CairoMakie.Reverse(:berlin))

        # Map values -> colors from the heatmap colormap
        cmap = cgrad(:berlin, 256)
        vminl, vmaxl = extrema(cm_liq)
        vminf, vmaxf = extrema(cm_flu)
        norm(v, vmin, vmax) = (v - vmin) / (vmax - vmin + eps())  # normalize into [0,1]

        for (coord, lbl, val) in zip(coords, cm_lab, vec(cm_liq))
            # get RGBA color for this value
            c = cmap[Int(clamp(round(Int, norm(val, vminl, vmaxl)*255)+1, 1, 256))]
            # compute perceived luminance (Y from Rec.601)
            lum = 0.299*c.r + 0.587*c.g + 0.114*c.b
            txtcolor = lum > 0.5 ? :black : :white
            text!(ax5, coord[1], coord[2],
                text = "$(lbl) = $(round(val)) %",
                align = (:center, :center), color = txtcolor, fontsize = 10)
        end
        for (coord, lbl, val) in zip(coords, cm_lab, vec(cm_flu))
            # get RGBA color for this value
            c = cmap[Int(clamp(round(Int, norm(val, vminf, vmaxf)*255)+1, 1, 256))]
            # compute perceived luminance (Y from Rec.601)
            lum = 0.299*c.r + 0.587*c.g + 0.114*c.b
            txtcolor = lum > 0.5 ? :black : :white
            text!(ax6, coord[1], coord[2],
                text = "$(lbl) = $(round(val)) %",
                align = (:center, :center), color = txtcolor, fontsize = 10)
        end

        # Make the labels look nice
        ax1.limits = (0.5, 3.5, -1.0, 2.0)
        ax1.xticks = ([1,2,3], [L"SiO$_2$", L"Al$_2$O$_3$", L"$$CaO"])
        ax1.xticklabelrotation = π/4
        text!(ax1, "a", position = (0.05, 0.85), space = :relative)
        ax2.limits = (0.5, 5.5, -0.5, 1.0)
        ax2.xticks = ([1,2,3,4,5], [L"$$MgO", L"$$FeO", L"Na$_2$O + K$_2$O", L"TiO$_2$", L"H$_2$O"])
        ax2.xticklabelrotation = π/4
        text!(ax2, "d", position = (0.05, 0.85), space = :relative)
        ax3.xticks = ([1,2], [L"$\phi$ Liq", L"$\phi$ Flu",])
        ax3.limits = (0.5, 2.5, -1.0, 5.0)
        ax3.xticklabelrotation = π/4
        text!(ax3, "b", position = (0.05, 0.85), space = :relative)
        ax4.limits = (0.5, 3.5, -10, 30)
        ax4.xticks = ([1,2,3], [L"$\rho$ Sys", L"$\rho$ Liq", L"$\rho$ Flu"])
        ax4.xticklabelrotation = π/4
        text!(ax4, "e", position = (0.05, 0.85), space = :relative)
        text!(ax5, "c", position = (0.05, 0.85), space = :relative)
        text!(ax6, "f", position = (0.05, 0.85), space = :relative)
        hidedecorations!(ax5)
        hidedecorations!(ax6)

        display(fg2)
        # Save the figure
        if !isdir("./user/figures")
            mkpath("./user/figures")
        end
        save("./user/figures/Figure_10_MLP_errors.png", fg2, px_per_unit = 4)
    end

    # Return
    return nothing
end

# Confusion matrix
function confusion_matrix(y_true_prob::AbstractArray{<:Real},
                          y_pred_prob::AbstractArray{<:Real};
                          τ::Real = 0.5)
    y_true = y_true_prob .>= τ
    y_pred = y_pred_prob .>= τ

    TP = count(y_true .& y_pred)
    FP = count(.!y_true .& y_pred)
    FN = count(y_true .& .!y_pred)
    TN = count(.!y_true .& .!y_pred)

    return [TP FP; FN TN]
end

# Reduced-input GRD08 viscosity (Giordano et al., 2008) with Na:K allocation parameter.
# Dependencies: JSON not needed; Plots.jl optional for plotting
# ] add Plots

"""
    viscosity_GRD08_reduced(
        wt::AbstractVector{<:Real};
        temp_C::Real,
        alpha_Na::Real=0.5,
        do_plot::Bool=true
    ) -> (; viscosity_Pa_s, log10_visc, A, B, C, Tg, alpha_Na)

Compute melt viscosity (Pa·s) using the GRD08 VFT model from a reduced oxide set.

Inputs
------
- `wt`: weight-% vector in this exact order
    [SiO2, Al2O3, CaO, MgO, FeO, (Na2O+K2O), TiO2, H2O]
- `temp_C`: temperature in °C
- `alpha_Na`: fraction of total alkalis assigned to Na2O in mole fraction space (0 ≤ α ≤ 1).
              Only the B₆ term depends on Na specifically; other terms use (Na2O + K2O).
- `do_plot`: if true, plots log10(viscosity) vs 10000/T(K) for 700–1399 °C.

Assumptions / limitations
-------------------------
- Unavailable oxides are set to zero: MnO=0, P2O5=0, F2O_1=0, Fe2O3=0 ⇒ FeOtot=FeO.
- This preserves the GRD08 functional form; the only approximation is allocating (Na+K)
  via `alpha_Na` for the B₆ term. The C-terms and all other B-terms using alkalis as a sum
  remain exact under this reduction.

Returns
-------
NamedTuple with viscosity (Pa·s), log10(viscosity), VFT A,B,C, Tg, and alpha_Na used.
"""
function viscosity_GRD08_reduced(
    wt::AbstractVector{<:Real};
    temp_C::Real,
    alpha_Na::Real=0.5,
    do_plot::Bool=true
)
    @assert length(wt) == 8 "wt must have 8 entries: [SiO2, Al2O3, CaO, MgO, FeO, (Na2O+K2O), TiO2, H2O]"
    @assert 0.0 <= alpha_Na <= 1.0 "alpha_Na must be in [0,1]"

    # Unpack input (wt%)
    wt_SiO2, wt_Al2O3, wt_CaO, wt_MgO, wt_FeO, wt_Alk, wt_TiO2, wt_H2O = Float64.(wt)

    # Missing oxides set to zero
    wt_MnO   = 0.0
    wt_P2O5  = 0.0
    wt_F2O_1 = 0.0
    wt_Fe2O3 = 0.0

    # FeOtot = FeO + (Fe2O3 to FeO) but Fe2O3=0 here
    wt_FeOtot = wt_FeO

    # --- Normalization (anhydrous basis like original code) ---
    # Sum of anhydrous oxides present (excluding H2O; F=0 here)
    sum_anhyd = reduce(Base.:+, (
        wt_SiO2, wt_TiO2, wt_Al2O3, wt_FeOtot, wt_MnO,
        wt_MgO, wt_CaO, wt_Alk, wt_P2O5
    ))

    factor = (100.0 - wt_H2O) / (sum_anhyd + wt_F2O_1)  # wt_F2O_1 = 0 here

    # Normalized (anhydrous) wt%
    wtn_SiO2  = wt_SiO2  * factor
    wtn_TiO2  = wt_TiO2  * factor
    wtn_Al2O3 = wt_Al2O3 * factor
    wtn_FeOtot= wt_FeOtot* factor
    wtn_MnO   = wt_MnO   * factor
    wtn_MgO   = wt_MgO   * factor
    wtn_CaO   = wt_CaO   * factor
    wtn_Alk   = wt_Alk   * factor
    wtn_P2O5  = wt_P2O5  * factor
    wtn_F2O_1 = wt_F2O_1 * factor  # = 0

    # Allocate Na vs K only where needed (B6 term uses Na specifically)
    wtn_Na2O  = alpha_Na * wtn_Alk
    wtn_K2O   = (1.0 - alpha_Na) * wtn_Alk

    # --- Convert to moles ---
    MM = Dict(
        "SiO2"  => 60.0843,
        "TiO2"  => 79.8658,
        "Al2O3" => 101.961276,
        "FeO"   => 71.8444,
        "MnO"   => 70.937449,
        "MgO"   => 40.3044,
        "CaO"   => 56.0774,
        "Na2O"  => 61.97894,
        "K2O"   => 94.196,
        "P2O5"  => 141.9446,
        "H2O"   => 18.01528,
        "F2O_1" => 37.9968,
    )

    # [SiO2, TiO2, Al2O3, FeOtot, MnO, MgO, CaO, Na2O, K2O, P2O5, H2O, F2O_1]
    moles = [
        wtn_SiO2   / MM["SiO2"],
        wtn_TiO2   / MM["TiO2"],
        wtn_Al2O3  / MM["Al2O3"],
        wtn_FeOtot / MM["FeO"],
        wtn_MnO    / MM["MnO"],
        wtn_MgO    / MM["MgO"],
        wtn_CaO    / MM["CaO"],
        wtn_Na2O   / MM["Na2O"],
        wtn_K2O    / MM["K2O"],
        wtn_P2O5   / MM["P2O5"],
        wt_H2O     / MM["H2O"],     # unnormalized H2O per original code
        wtn_F2O_1  / MM["F2O_1"],
    ]

    total_moles = sum(moles)
    mf = moles .* (100.0 / total_moles)  # mole %
    mf_SiO2, mf_TiO2, mf_Al2O3, mf_FeOtot, mf_MnO, mf_MgO,
    mf_CaO, mf_Na2O, mf_K2O, mf_P2O5, mf_H2O, mf_F2O_1 = mf

    # --- GRD08 fitting parameters ---
    A = -4.55
    B1,  B2,   B3,   B4,   B5,    B6,    B7   = 159.560, -173.340, 72.130, 75.690, -38.980, -84.080, 141.540
    B11, B12,  B13                         =  -2.43,     -0.91,     17.62
    C1,  C2,   C3,   C4,   C5,    C6,    C11  =   2.75,    15.720,   8.320, 10.20, -12.290, -99.540, 0.300

    # --- Compute B, C ---
    B = (mf_SiO2 + mf_TiO2) * B1 +
        mf_Al2O3 * B2 +
        (mf_FeOtot + mf_MnO + mf_P2O5) * B3 +
        mf_MgO * B4 +
        mf_CaO * B5 +
        (mf_Na2O + mf_H2O + mf_F2O_1) * B6 +  # Na-specific here
        ((mf_H2O + mf_F2O_1) + log(1.0 + mf_H2O)) * B7 +
        (mf_SiO2 + mf_TiO2) * (mf_FeOtot + mf_MnO + mf_MgO) * B11 +
        (mf_SiO2 + mf_TiO2 + mf_Al2O3 + mf_P2O5) * (mf_Na2O + mf_K2O + mf_H2O) * B12 +
        mf_Al2O3 * (mf_Na2O + mf_K2O) * B13

    C =  mf_SiO2 * C1 +
         (mf_TiO2 + mf_Al2O3) * C2 +
         (mf_FeOtot + mf_MnO + mf_MgO) * C3 +
          mf_CaO * C4 +
         (mf_Na2O + mf_K2O) * C5 +
          log(1.0 + mf_H2O) * C6 +
         (mf_Al2O3 + mf_FeOtot + mf_MnO + mf_MgO + mf_CaO - mf_P2O5) *
         (mf_Na2O + mf_K2O + mf_H2O) * C11

    Tg = B / (12 - A) + C
    log10_visc = A + B / (temp_C + 273.15 - C)
    viscosity = 10.0 ^ log10_visc

    # @printf("A = %3.3f\n", A)
    # @printf("B = %3.3f\n", B)
    # @printf("C = %3.3f\n", C)
    # @printf("Tg = %3.3f K\n", Tg)
    # @printf("viscosity = %3.3f Pa·s (log10(visc)=%3.3f) at T = %3.3f °C (alpha_Na=%.2f)\n",
    #         viscosity, log10_visc, temp_C, alpha_Na)

    if do_plot
        temps = 700:1399
        invT = [10000.0 / (t + 273.15) for t in temps]
        logvis = [A + B / (t + 273.15 - C) for t in temps]
        plot(invT, logvis, label="GRD08 (reduced input)", xlabel="10000 / T (K)", ylabel="log viscosity (Pa·s)")
        display(current())
    end

    return (; viscosity_Pa_s=viscosity,
            log10_visc=log10_visc, A=A, B=B, C=C, Tg=Tg, alpha_Na=alpha_Na)
end

# Example usage:
# wt = [50.0, 15.5, 10.0, 7.0, 8.0, 5.5, 1.0, 3.0]  # [SiO2, Al2O3, CaO, MgO, FeO, (Na2O+K2O), TiO2, H2O]
# out = viscosity_GRD08_reduced(wt; temp_C=1200.0, alpha_Na=0.6, do_plot=true)

axs = evaluate_results();