# Original file name in my system: ex_evaluate_crossValidation_paper_v4.jl
using JLD2
using Lux, Statistics, Impute, ROC, Random
using MAGEMin_C
using CairoMakie, Base, MathTeXEngine, ColorSchemes
include("MAGEMin_MLPs.jl")
using .MAGEMin_MLPs

# Tolerances
const ϵ_log = 1e-5
DatType = Float32

@views function evaluate_results()

    # Randomness
    seed = 123
    rng  = Random.MersenneTwister()
    Random.seed!(rng, seed)

    # Set data path
    data_dir = "/media/largeData/ArcMagma_NeuralNetworks/MLP/NeuralNetworks/CrossValidation_062025/"
    dirs     = readdir(data_dir) 
    mnames   = contains.(dirs, "Hyper") |> idx -> dirs[idx .== 1]

    # Set vars
    nfeat = 8
    nfold = 5

    # Initialize data Matrix
    datmat = Matrix{Float64}(undef, length(mnames), nfeat)

    # Loop through models
    for (idx_model, mname) in enumerate(mnames)

        # Extract model info
        minfo = JLD2.load("$data_dir/$mname", "Info")
        nlay  = split(mname, "_") |> splts -> split(splts[3], "H") |> nlay -> parse(Int64, nlay[1])

        # Store cross validation independent model info
        datmat[idx_model, 1] = nlay
        datmat[idx_model, 2] = minfo["nNeurons"]
        datmat[idx_model, 3] = minfo["batchsize"]

        # Loop through folds of current hyperparameter configuration
        vloss_avg = 0.0; vloss_min = 1e4; vloss_max = 0.0; MAE = 0.0; overfit = 0
        for k in 1:nfold
            model = JLD2.load("$data_dir/$mname", "Fold_$k/MLP/model")
            stats = JLD2.load("$data_dir/$mname", "Fold_$k/Stats")
            
            # Min, max, average loss
            vloss_avg += stats["val_losses"][end] / nfold
            vloss_min  = min(vloss_min, stats["val_losses"][end])
            vloss_max  = max(vloss_min, stats["val_losses"][end])

            # Overfitting
            overfit += stats["overfit"][k]

            # Load validation data and evaluate MAE
            in_val   = JLD2.load("$data_dir/$mname", "Fold_$k/Data/input_val")
            out_val  = JLD2.load("$data_dir/$mname", "Fold_$k/Data/output_val")
            in_mean  = JLD2.load("$data_dir/$mname", "Fold_$k/Scaling/mean_input_train")
            in_std   = JLD2.load("$data_dir/$mname", "Fold_$k/Scaling/std_input_train")
            out_mean = JLD2.load("$data_dir/$mname", "Fold_$k/Scaling/mean_output_train")
            out_std  = JLD2.load("$data_dir/$mname", "Fold_$k/Scaling/std_output_train")
            states_t = Lux.testmode(model.st)
            in_val_sc = MAGEMin_MLPs.TransformData(in_val, transform = :zscore, dims = 2, scale = true, mean_data = in_mean, std_data = in_std; DatType = DatType)
            states_t = Lux.testmode(model.st)
            (p̂1, p̂2, ŷ1, ŷ2, ŷ3, ŷ4, ŷ5, ŷ6) = Lux.apply(model.model.u, DatType.(in_val_sc), model.ps, states_t)[1]
            ŷ_sc = vcat(p̂1, vcat(p̂2, vcat(ŷ1, vcat(ŷ2, vcat(ŷ3, vcat(ŷ4, vcat(ŷ5, ŷ6)))))))
            ŷ  = MAGEMin_MLPs.TransformDataBack(ŷ_sc; transform = :zscore, dims=2, mean_data = out_mean, std_data = out_std)
            ŷ[[5, 7, 8, 9], :] .= exp.(ŷ[[5, 7, 8, 9], :]) .- DatType(ϵ_log)
            # Show error
            mask_liq = ŷ[1, :] .> 0.5
            mask_flu = ŷ[2, :] .> 0.5
            ŷ[3:10, :] ./= sum(ŷ[3:10, :], dims = 1)
            for idx in 3:10
                ŷ[idx, :] .*= mask_liq
            end
            ŷ[12, :] .*= mask_liq
            ŷ[14, :] .*= mask_liq
            ŷ[13, :] .*= mask_flu
            ŷ[15, :] .*= mask_flu
            # MAE += mean(sum(abs.(out_val .- ŷ), dims = 1) ./ nfold, dims = 2)[1]
            MAE += sum(mean(abs.(out_val .- ŷ), dims = 2), dims = 1)[1] / nfold
        end

        # Store cross validation dependent model info
        overfit > 0 ? datmat[idx_model, nfeat] = 0.5 : datmat[idx_model, nfeat] = 1.0
        datmat[idx_model, 4] = vloss_min
        datmat[idx_model, 5] = vloss_max
        datmat[idx_model, 6] = vloss_avg
        datmat[idx_model, 7] = MAE
    end

    # Figure settings
    fwi      = 140.0
    fhe      = 140.0
    dpi      = 300.0
    ftsize   = 24.0
    ncheck   = 20
    save_fig = true
    nNeurons = [64, 128, 256]
    nLayers  = [1, 3, 5]
    leg_elems = [
        LineElement(color = :gray, linestyle = :solid),
        LineElement(color = :gray, linestyle = :dash),
        LineElement(color = :gray, linestyle = :dot),
    ]
    leg_str = string.(nNeurons)

    # Initiaize figure
    fsize = (fwi, fhe) .* dpi ./ 25.4
    f     = Figure(size = fsize, fontsize = ftsize)
    gl1   = GridLayout(f[1, 1:3])
    gl2   = GridLayout(f[2, 1:3])
    gl3   = GridLayout(f[3, 1:3])
    cmap  = cgrad(:bamako, [1, 2, 3] ./ 3.0; alpha = 1.0, rev = false, categorical = true)
    αs    = [3.0, 2.0, 1.0] ./ 3.0
    lstyle = [:solid, :dash, :dot]
    ax1   = Axis(
        gl1[1,1], 
        aspect = 1.0,
        xlabel = L"$$Layers [ ]",
        ylabel = L"$$Neurons [ ]",
        title = L"$$batch size = 1024",
        xticks = nLayers,
        yticks = nNeurons
    )
    ax2   = Axis(
        gl1[1,2],
        aspect = 1.0,
        xlabel = L"$$Layers [ ]",
        title = L"$$batch size = 4096",
        xticks = nLayers,
        yticks = nNeurons
    )
    ax3   = Axis(
        gl1[1,3],
        aspect = 1.0,
        xlabel = L"$$Layers [ ]",
        title = L"$$batch size = 16384",
        xticks = nLayers,
        yticks = nNeurons
    )
    ax4   = Axis(
        gl2[1,1],
        aspect = 1.0,
        xlabel = L"$$Layers [ ]",
        ylabel = L"$$Neurons [ ]",
        xticks = nLayers,
        yticks = nNeurons
    )
    ax5   = Axis(
        gl2[1,2],
        aspect = 1.0,
        xlabel = L"$$Layers [ ]",
        xticks = nLayers,
        yticks = nNeurons
    )
    ax6   = Axis(
        gl2[1,3],
        aspect = 1.0,
        xlabel = L"$$Layers [ ]",
        xticks = nLayers,
        yticks = nNeurons
    )
    ax7   = Axis(
        gl3[1,1:3],
        xlabel = L"$$Epochs [ ]",
        ylabel = L"mean($\mathcal{L}$) [ ]",
        yscale = log10,
        ytickformat = "{:.2f}",
        yminorticksvisible = true,
        yminorgridvisible = true,
        yminorticks = [
               1e-2 * i for i in 2:9
           ] ∪ [
               1e-1 * i for i in 2:9
           ] ∪ [
               1 * i for i in 2:9
           ]
    )

    # Prepare data for plotting
    x_layers     = Float64.(nLayers)
    y_neurons    = Float64.(nNeurons)
    datmat_1024  = findall(x -> x ==  1024.0, datmat[:, 3]) |> idx -> datmat[idx, :]
    datmat_4096  = findall(x -> x ==  4096.0, datmat[:, 3]) |> idx -> datmat[idx, :]
    datmat_16384 = findall(x -> x == 16384.0, datmat[:, 3]) |> idx -> datmat[idx, :]
    vloss_avg_1024  = zeros(Float64, length(y_neurons), length(x_layers))
    vloss_avg_4096  = zeros(Float64, length(y_neurons), length(x_layers))
    vloss_avg_16384 = zeros(Float64, length(y_neurons), length(x_layers))
    overfit_1024    = zeros(Float64, length(y_neurons), length(x_layers))
    overfit_4096    = zeros(Float64, length(y_neurons), length(x_layers))
    overfit_16384   = zeros(Float64, length(y_neurons), length(x_layers))
    MAE_1024        = zeros(Float64, length(y_neurons), length(x_layers))
    MAE_4096        = zeros(Float64, length(y_neurons), length(x_layers))
    MAE_16384       = zeros(Float64, length(y_neurons), length(x_layers))
    for idx in axes(datmat_1024, 1)
        idx_x = findfirst(x -> x == datmat_1024[idx, 1], x_layers)
        idx_y = findfirst(y -> y == datmat_1024[idx, 2], y_neurons)
        vloss_avg_1024[idx_x,  idx_y] = datmat_1024[idx,  6]
        MAE_1024[idx_x,    idx_y] = datmat_1024[idx,  7]
        overfit_1024[idx_x,    idx_y] = datmat_1024[idx,  nfeat]
    end
    for idx in axes(datmat_4096, 1)
        idx_x = findfirst(x -> x == datmat_4096[idx, 1], x_layers)
        idx_y = findfirst(y -> y == datmat_4096[idx, 2], y_neurons)
        vloss_avg_4096[idx_x,  idx_y] = datmat_4096[idx,  6]
        MAE_4096[idx_x,    idx_y] = datmat_4096[idx,  7]
        overfit_4096[idx_x,    idx_y] = datmat_4096[idx,  nfeat]
    end
    for idx in axes(datmat_16384, 1)
        idx_x = findfirst(x -> x == datmat_16384[idx, 1], x_layers)
        idx_y = findfirst(y -> y == datmat_16384[idx, 2], y_neurons)
        vloss_avg_16384[idx_x,  idx_y] = datmat_16384[idx,  6]
        MAE_16384[idx_x,    idx_y] = datmat_16384[idx,  7]
        overfit_16384[idx_x,    idx_y] = datmat_16384[idx,  nfeat]
    end

    # Loop through models
    for (_, mname) in enumerate(mnames)

        # Extract model info
        nlay  = split(mname, "_") |> splts -> split(splts[3], "H") |> nlay -> parse(Int64, nlay[1])
        nneu  = split(mname, "_") |> splts -> split(splts[2], "H") |> nneu -> parse(Int64, nneu[1])

        # Loop through folds of current hyperparameter configuration
        stats = JLD2.load("$data_dir/$mname", "Fold_1/Stats")
        val_loss = [0.0 for _ in eachindex(stats["val_losses"][1:end-1])]
        tra_loss = [0.0 for _ in eachindex(stats["train_losses"][1:end-1])]
        overfit = 0
        for k in 1:nfold
            stats = JLD2.load("$data_dir/$mname", "Fold_$k/Stats")
            stats_cnt_val = [0.0 for _ in eachindex(val_loss)]
            stats_cnt_tra = [0.0 for _ in eachindex(tra_loss)]
            stats_cnt_val[1:size(stats["val_losses"][1:end-1],   1)] .= stats["val_losses"][1:end-1]
            stats_cnt_tra[1:size(stats["train_losses"][1:end-1], 1)] .= stats["train_losses"][1:end-1]
            val_loss .+= stats_cnt_val
            tra_loss .+= stats_cnt_tra
            overfit += stats["overfit"][k]
        end
        val_loss ./= nfold
        tra_loss ./= nfold
        idxl = findfirst(x -> x == nlay, nLayers)
        idxn = findfirst(x -> x == nneu, nNeurons)

        # Plot loss
        linewidth = 3.0
        overfit > 0 ? linewidth = 0.5 : nothing
        if mname == "HyperParameterStudy_256HD_3HL_16384batchsize_500epochs_crossvalMLP.jld2"
            lines!(ax7,vec(1:length(val_loss)) .* ncheck, val_loss, color = :skyblue1, linestyle = :dash,  linewidth = linewidth, label = L"$$Validation")
            lines!(ax7,vec(1:length(tra_loss)) .* ncheck, tra_loss, color = :skyblue4, linestyle = :solid, linewidth = linewidth, label = L"$$Training")
            axislegend(ax7, L"$$3 Layers, 256 Neurons, 16384 batch size", backgroundcolor = (:white, 0.0), framevisible = false)
        end
    end

    # Fill subplots
    crange_loss = (0.0, 0.5)
    MAE_all = hcat(MAE_16384, hcat(MAE_4096, MAE_1024))
    display(MAE_all)
    # minMAE = min(min(minimum(MAE_16384), minimum(MAE_4096)), minimum(MAE_1024))
    # maxMAE = max(max(maximum(MAE_16384), maximum(MAE_4096)), maximum(MAE_1024))
    minMAE = minimum(MAE_all)
    maxMAE = maximum(MAE_all)
    display([minMAE, maxMAE])
    hm1 = contourf!(ax1, vloss_avg_1024', colormap = :bilbao)
    hm2 = contourf!(ax2, vloss_avg_4096', colormap = :bilbao)
    hm3 = contourf!(ax3, vloss_avg_16384', colormap = :bilbao)
    hm4 = heatmap!(ax4, MAE_1024', colormap  = :acton, colorrange = (minMAE, maxMAE))
    hm5 = heatmap!(ax5, MAE_4096', colormap  = :acton, colorrange = (minMAE, maxMAE))
    hm6 = heatmap!(ax6, MAE_16384', colormap = :acton, colorrange = (minMAE, maxMAE))
    sc1 = scatter!(ax6, 2, 3, marker = :star8, markersize = 30, color = :white)

    ax1.xticks = (1:3, string.(nLayers))
    ax1.yticks = (1:3, string.(nNeurons))
    ax2.xticks = (1:3, string.(nLayers))
    ax2.yticks = (1:3, string.(nNeurons))
    ax3.xticks = (1:3, string.(nLayers))
    ax3.yticks = (1:3, string.(nNeurons))
    ax4.xticks = (1:3, string.(nLayers))
    ax4.yticks = (1:3, string.(nNeurons))
    ax5.xticks = (1:3, string.(nLayers))
    ax5.yticks = (1:3, string.(nNeurons))
    ax6.xticks = (1:3, string.(nLayers))
    ax6.yticks = (1:3, string.(nNeurons))
    ax7.limits = (ncheck, 500, 0.01, 5.0)

    # Add colorbars
    cb1 = Colorbar(gl1[1, 4], hm1, label = L"mean($\mathcal{L}_{val}$) [ ]", tellheight = false)
    cb2 = Colorbar(gl2[1, 4], hm4, label = L"$$MAE [ ]", tellheight = false)
    cb2.colorrange = (minMAE, maxMAE)
    
    # Add subplot labels
    text!(ax1, "a", position = (0.05, 0.9), space = :relative, color = :white)
    text!(ax2, "b", position = (0.05, 0.9), space = :relative, color = :white)
    text!(ax3, "c", position = (0.05, 0.9), space = :relative, color = :white)
    text!(ax4, "d", position = (0.05, 0.9), space = :relative, color = :white)
    text!(ax5, "e", position = (0.05, 0.9), space = :relative, color = :white)
    text!(ax6, "f", position = (0.05, 0.9), space = :relative, color = :white)
    text!(ax7, "g", position = (0.05, 0.9), space = :relative)
    
    # Display and/or save figure
    display(f)
    save_fig ? save("/home/lcandiot/Developer/MeltMigration_MagmaticSystems/paper_figures_data/png/Figure_3_MLP_crossval_results.png", f, px_per_unit = 1) : nothing

    # Return
    return nothing
end

axs = evaluate_results();