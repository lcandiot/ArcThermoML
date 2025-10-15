using JLD2
using Lux, Random, Zygote, Optimisers, Statistics, MLUtils
using Printf
using .MAGEMin_MLPs

# Define Lux structure
struct MLP{U} <: Lux.AbstractLuxWrapperLayer{:u}
    u::U
end

# Set tolerances
const ϵ_log = 1e-5

@views function create_mlp(
    hidden_dims :: Int,
    hidden_lays :: Int
    )

    # Backbone
    if hidden_lays == 1
        backbone = Lux.Chain(
        Lux.Dense(10          => hidden_dims, tanh; init_weight = Lux.glorot_normal, init_bias = Lux.zeros32); name = "backbone"
        )
    elseif hidden_lays == 2
        backbone = Lux.Chain(
        Lux.Dense(10          => hidden_dims, tanh; init_weight = Lux.glorot_normal, init_bias = Lux.zeros32),
        Lux.Dense(hidden_dims => hidden_dims, tanh; init_weight = Lux.glorot_normal, init_bias = Lux.zeros32); name = "backbone"
        )
    elseif hidden_lays == 3
        backbone = Lux.Chain(
        Lux.Dense(10          => hidden_dims, tanh; init_weight = Lux.glorot_normal, init_bias = Lux.zeros32),
        Lux.Dense(hidden_dims => hidden_dims, tanh; init_weight = Lux.glorot_normal, init_bias = Lux.zeros32),
        Lux.Dense(hidden_dims => hidden_dims, tanh; init_weight = Lux.glorot_normal, init_bias = Lux.zeros32); name = "backbone"
        )
    elseif hidden_lays == 4
        backbone = Lux.Chain(
        Lux.Dense(10          => hidden_dims, tanh; init_weight = Lux.glorot_normal, init_bias = Lux.zeros32),
        Lux.Dense(hidden_dims => hidden_dims, tanh; init_weight = Lux.glorot_normal, init_bias = Lux.zeros32),
        Lux.Dense(hidden_dims => hidden_dims, tanh; init_weight = Lux.glorot_normal, init_bias = Lux.zeros32),
        Lux.Dense(hidden_dims => hidden_dims, tanh; init_weight = Lux.glorot_normal, init_bias = Lux.zeros32); name = "backbone"
        )
    elseif hidden_lays == 5
        backbone = Lux.Chain(
        Lux.Dense(10          => hidden_dims, tanh; init_weight = Lux.glorot_normal, init_bias = Lux.zeros32),
        Lux.Dense(hidden_dims => hidden_dims, tanh; init_weight = Lux.glorot_normal, init_bias = Lux.zeros32),
        Lux.Dense(hidden_dims => hidden_dims, tanh; init_weight = Lux.glorot_normal, init_bias = Lux.zeros32),
        Lux.Dense(hidden_dims => hidden_dims, tanh; init_weight = Lux.glorot_normal, init_bias = Lux.zeros32),
        Lux.Dense(hidden_dims => hidden_dims, tanh; init_weight = Lux.glorot_normal, init_bias = Lux.zeros32); name = "backbone"
        )
    end

    # Heads
    liqClass_head = Lux.Chain(
    Lux.Dense(hidden_dims => 1, sigmoid; init_weight = Lux.glorot_normal, init_bias = Lux.zeros32);
    name = "liqClass_head"
    )
    flClass_head = Lux.Chain(
    Lux.Dense(hidden_dims => 1, sigmoid; init_weight = Lux.glorot_normal, init_bias = Lux.zeros32);
    name = "flClass_head"
    )
    majOx_Head = Lux.Chain(
        Lux.Dense(hidden_dims => 2; init_weight = Lux.glorot_normal, init_bias = Lux.zeros32);
        name = "majOx_Head"
    )
    midOx_Head = Lux.Chain(
        Lux.Dense(hidden_dims => 3; init_weight = Lux.glorot_normal, init_bias = Lux.zeros32);
        name = "midOx_Head"
    )
    minOx_Head = Lux.Chain(
        Lux.Dense(hidden_dims => 3; init_weight = Lux.glorot_normal, init_bias = Lux.zeros32);
        name = "minOx_Head"
    )
    sys_Head = Lux.Chain(
        Lux.Dense(hidden_dims => 1; init_weight = Lux.glorot_normal, init_bias = Lux.zeros32); name = "sys_Head"
    )
    fra_Head = Lux.Chain(
            Lux.Dense(hidden_dims => 2; init_weight = Lux.glorot_normal, init_bias = Lux.zeros32); name = "fra_Head"
    )
    rho_Head = Lux.Chain(
        Lux.Dense(hidden_dims => 2; init_weight = Lux.glorot_normal, init_bias = Lux.zeros32); name = "rho_Head"
    )

    # Connect
    return Lux.Chain(
        backbone,
        Lux.BranchLayer(
            liqClass_head,
            flClass_head,
            majOx_Head,
            midOx_Head,
            minOx_Head,
            sys_Head,
            fra_Head,
            rho_Head
        )
    )
end

@views function MLP(;
    hidden_dims :: Int=32,
    hidden_lays :: Int=0
)
    return MLP(
        create_mlp(hidden_dims, hidden_lays)
    )
end

"""
```
    early_stopping!(state, x_test, y_test, test_loss, counter)
```

Evaluates the deterioration of the test loss. Returns `counter` which is the number of consecutive epochs during which the test loss has deteriorated and `loss` which is the test loss.

# Arguments
* `state` : Training state of `Lux` model
* `x_test` : Test fraction of input data
* `y_test` : Test fraction of output data
* `test_loss` : Test loss
* `counter` : Deterioration counter
"""
@views function early_stopping!(
    train_st,
    x,
    y,
    val_loss,
    counter,
)

    cdev = cpu_device()

    # Make a prediction for the validation data
    test_st = Lux.testmode(train_st.states)
    (p̂1, p̂2, ŷ1, ŷ2, ŷ3, ŷ4, ŷ5, ŷ6) = cdev(Lux.apply(train_st.model.u, x, train_st.parameters, test_st)[1])

    # Individual losses
    p̂1_loss  = Lux.BinaryCrossEntropyLoss(; epsilon = 1e-8)(p̂1', y[1, :])
    p̂2_loss  = Lux.BinaryCrossEntropyLoss(; epsilon = 1e-8)(p̂2', y[2, :])
    ŷ1a_loss = Lux.HuberLoss()((y[1,:] .== 1) .* ŷ1[1, :], (y[1,:] .== 1) .* y[3,  :] )
    ŷ1b_loss = Lux.HuberLoss()((y[1,:] .== 1) .* ŷ1[2, :], (y[1,:] .== 1) .* y[4,  :] )
    ŷ2a_loss = Lux.HuberLoss()((y[1,:] .== 1) .* ŷ2[1, :], (y[1,:] .== 1) .* y[5,  :] )
    ŷ2b_loss = Lux.HuberLoss()((y[1,:] .== 1) .* ŷ2[2, :], (y[1,:] .== 1) .* y[6,  :] )
    ŷ2c_loss = Lux.HuberLoss()((y[1,:] .== 1) .* ŷ2[3, :], (y[1,:] .== 1) .* y[7,  :] )
    ŷ3a_loss = Lux.HuberLoss()((y[1,:] .== 1) .* ŷ3[1, :], (y[1,:] .== 1) .* y[8,  :] )
    ŷ3b_loss = Lux.HuberLoss()((y[1,:] .== 1) .* ŷ3[2, :], (y[1,:] .== 1) .* y[9,  :] )
    ŷ3c_loss = Lux.HuberLoss()((y[1,:] .== 1) .* ŷ3[3, :], (y[1,:] .== 1) .* y[10, :] )
    ŷ4_loss  = Lux.HuberLoss()(                  ŷ4,                         y[11, :]')
    ŷ5a_loss = Lux.HuberLoss()((y[1,:] .== 1) .* ŷ5[1, :], (y[1,:] .== 1) .* y[12, :] )
    ŷ5b_loss = Lux.HuberLoss()((y[2,:] .== 1) .* ŷ5[2, :], (y[2,:] .== 1) .* y[13, :] )
    ŷ6a_loss = Lux.HuberLoss()((y[1,:] .== 1) .* ŷ6[1, :], (y[1,:] .== 1) .* y[14, :] )
    ŷ6b_loss = Lux.HuberLoss()((y[2,:] .== 1) .* ŷ6[2, :], (y[2,:] .== 1) .* y[15, :] )

    # Sum up losses
    loss = p̂1_loss + p̂2_loss + (ŷ1a_loss + ŷ1b_loss) + (ŷ2a_loss + ŷ2b_loss + ŷ2c_loss) + (ŷ3a_loss + ŷ3b_loss + ŷ3c_loss) + ŷ4_loss + (ŷ5a_loss + ŷ5b_loss) + (ŷ6a_loss + ŷ6b_loss)

    if ( loss ) >= val_loss
        counter += 1
    else
        counter = 0
    end

    # Return
    return counter, loss
end

"""
```
    train_model_crossval(x, y, dev; kwargs)
```

Trains the neural network with cross validation of the data set. Split between training and testing data is done internally.

# Arguments
* `x` : Input data
* `y` : Output data
* `dev` : Initialized device (CPU/GPU)
* `rng` : Random number generator
* `idx_split` : Named Tuple containing the inidices for training, `idx_train`, and testing, `idx_test`, data split

# Keyword arguments
* `seed` : Seed for the random number generator
* `nepochs` : No. of epochs to train
* `hidden_dims` : No. neurons per hidden layer
* `early_stop` : Exit criterion for early stopping
* `nprint` : Progress printing interval
* `nFolds` : No. cross validation iterations
* `save_model` : Save the trained model at each fold
* `fname_root` : Root for the file name. Used to store the trained model at each fold
* `log_transform` : Switch to activate log transformation
"""
@views function train_model_crossval(
    x             :: AbstractArray,
    y             :: AbstractArray,
    dev,
    rng;
    nEpochs       :: Int = 5_00,
    hidden_dims   :: Int = 32,
    hidden_lays   :: Int = 1,
    hyperparams   :: Dict{String, Real},
    early_stop    :: Int = 10,
    nprint        :: Int = 20,
    nFolds        :: Int = 5,
    save_model    :: Bool = false,
    fname_root    :: String,
    log_transform :: Bool = false,
    DatType       :: Type = Float32
)
    # Set CPU device
    cdev = cpu_device()

    # Set optimiser for model parts
    opt_backbone     = Optimisers.AdamW(1e-3, (0.9, 0.99), 1e-3, 1e-8)
    opt_liqClassHead = Optimisers.AdamW(1e-3, (0.9, 0.99), 1e-3, 1e-8)
    opt_fluClassHead = Optimisers.AdamW(1e-3, (0.9, 0.99), 1e-3, 1e-8)
    opt_majOxHead    = Optimisers.Adam(1e-3, (0.9, 0.99))
    opt_midOxHead    = Optimisers.Adam(1e-3, (0.9, 0.99))
    opt_minOxHead    = Optimisers.Adam(1e-3, (0.9, 0.99))
    opt_sys_Head     = Optimisers.Adam(1e-3, (0.9, 0.99))
    opt_fra_Head     = Optimisers.Adam(1e-3, (0.9, 0.99))
    opt_rho_Head     = Optimisers.Adam(1e-3, (0.9, 0.99))

    # Automatic differentiation
    vjp_rule = AutoZygote()

    # Split between training and validation set
    N                = size(x, 2)
    idx_tra, idx_val = MLUtils.kfolds(N, nFolds)

    # kFold validation loop
    overfit = [0 for _ in 1:nFolds]
    for k in 1:nFolds

        @printf "\nValidation %d / %d\n" k nFolds

        # Define the MLP structure
        mlp = MLP(; hidden_dims = hidden_dims, hidden_lays = hidden_lays)

        # Lux setup
        if DatType == Float64
            mlp = Lux.f64(mlp)
        end
        params, state = Lux.setup(rng, mlp) |> dev

        if DatType == Float64
            params, state = Lux.f64(params), Lux.f64(state)
        end

        # Create training state
        train_st = Training.TrainState(mlp, params, state, opt_backbone)

        # Create optimiser states
        opt_st_backbone     = Optimisers.setup(opt_backbone,     params[1]   ) |> dev
        opt_st_liqClassHead = Optimisers.setup(opt_liqClassHead, params[2][1]) |> dev
        opt_st_fluClassHead = Optimisers.setup(opt_fluClassHead, params[2][2]) |> dev
        opt_st_majOxHead    = Optimisers.setup(opt_majOxHead,    params[2][3]) |> dev
        opt_st_midOxHead    = Optimisers.setup(opt_midOxHead,    params[2][4]) |> dev
        opt_st_minOxHead    = Optimisers.setup(opt_minOxHead,    params[2][5]) |> dev
        opt_st_sys_Head      = Optimisers.setup(opt_sys_Head,    params[2][6]) |> dev
        opt_st_fra_Head     = Optimisers.setup(opt_fra_Head,     params[2][7]) |> dev
        opt_st_rho_Head     = Optimisers.setup(opt_rho_Head,     params[2][8]) |> dev

        x_tra = deepcopy(x[:, idx_tra[k]]) |> x -> DatType.(x)
        y_tra = deepcopy(y[:, idx_tra[k]]) |> x -> DatType.(x)
        x_val = deepcopy(x[:, idx_val[k]]) |> x -> DatType.(x)
        y_val = deepcopy(y[:, idx_val[k]]) |> x -> DatType.(x)

        @show intersect(eachcol(y_val), eachcol(y_tra))

        # Logit transform
        if log_transform
            x_tra[[2, 5, 6, 8, 9], :] .= log.(x_tra[[2, 5, 6, 8, 9], :] .+ DatType(ϵ_log))
            y_tra[[5, 7, 8, 9],    :] .= log.(y_tra[[5, 7, 8, 9],    :] .+ DatType(ϵ_log))
            x_val[[2, 5, 6, 8, 9], :] .= log.(x_val[[2, 5, 6, 8, 9], :] .+ DatType(ϵ_log))
            y_val[[5, 7, 8, 9],    :] .= log.(y_val[[5, 7, 8, 9],    :] .+ DatType(ϵ_log))
        end

        # Scale training data (only continuous part)
        mask_liq = y_tra[1, :] .> 0.0
        mask_flu = y_tra[2, :] .> 0.0
        mean_y   = zeros(DatType, size(y_tra, 1), 1)
        std_y    = zeros(DatType, size(y_tra, 1), 1)
        x_tra_sc, mean_x_tra, std_x_tra = MAGEMin_MLPs.TransformData(x_tra, transform = :zscore, dims = 2; DatType = DatType)
        _, mean_y_Xl, std_y_Xl          = MAGEMin_MLPs.TransformData(y_tra[3:10, mask_liq], transform = :zscore, dims = 2; DatType = DatType)
        _, mean_y_ρS, std_y_ρS          = MAGEMin_MLPs.TransformData(y_tra[11, :], transform = :zscore, dims = 1; DatType = DatType)
        _, mean_y_l,  std_y_l           = MAGEMin_MLPs.TransformData(y_tra[[12, 14], mask_liq], transform = :zscore, dims = 2; DatType = DatType)
        _, mean_y_f,  std_y_f           = MAGEMin_MLPs.TransformData(y_tra[[13, 15], mask_flu], transform = :zscore, dims = 2; DatType = DatType)

        # Add unscaled outputs
        mean_y[[1, 2],    1] .= 0.0
        mean_y[3:10,      1] .= mean_y_Xl
        mean_y[11,        1]  = mean_y_ρS[1]
        mean_y[[12, 14],  1] .= mean_y_l
        mean_y[[13, 15],  1] .= mean_y_f
        std_y[[1, 2],     1] .= 1.0
        std_y[3:10,       1] .= std_y_Xl
        std_y[11,         1]  = std_y_ρS[1]
        std_y[[12, 14],   1] .= std_y_l
        std_y[[13, 15],   1] .= std_y_f
        y_tra_sc = MAGEMin_MLPs.TransformData(y_tra, transform = :zscore, scale = true, dims = 2, mean_data = mean_y, std_data = std_y; DatType = DatType)
        dev(mean_y)
        dev(std_y)

        # Scale validation data to the mean and standard deviation of the training set
        x_val_sc = MAGEMin_MLPs.TransformData(x_val, transform = :zscore, dims = 2, scale = true, mean_data = cdev(mean_x_tra), std_data = cdev(std_x_tra), DatType = DatType) |> dev
        y_val_sc = MAGEMin_MLPs.TransformData(y_val, transform = :zscore, dims = 2, scale = true, mean_data = cdev(mean_y    ), std_data = cdev(std_y    ), DatType = DatType) |> dev

        # Revert logit transform for validation set
        if log_transform
             y_val[[5, 7, 8, 9], :] .= exp.(y_val[[5, 7, 8, 9], :]) .- DatType(ϵ_log)
        end

        # Create data loader
        data_loader = DataLoader((x_tra_sc, y_tra_sc); batchsize = hyperparams["bsize"], shuffle = false, rng = rng) |> dev

        # Initialize losses and counters
        es_counter = 0
        train_loss_array = []; val_loss_array = []; p̂1_loss_array = [] ; p̂2_loss_array = [] ; ŷ1_loss_array = []; ŷ2_loss_array = []; ŷ3_loss_array = []; ŷ4_loss_array = []; ŷ5_loss_array = []; ŷ6_loss_array = []
        precis_liq_array = []; recall_liq_array = []; precis_flu_array = []; recall_flu_array = []
        R∑Xl_array = []

        # Training loop
        val_loss = 10_000; 
        warmup_epochs = cld(nEpochs, 10)
        T_bb = 1; T_liqClass = 1; T_fluClass = 1; T_mOx = 1; T_tOx = 1; T_liq = 1; T_sys = 1; T_flu = 1; T_tot = nEpochs
        η_bb = 0.0; η_liqClass = 0.0; η_fluClass = 0.0; η_mOx = 0.0; η_tOx = 0.0; η_liq = 0.0; η_sys = 0.0; η_flu = 0.0
        precis_liq = 0.0; recall_liq = 0.0; precis_flu = 0.0; recall_flu = 0.0
        @printf "Starting training ...\n"
        for epoch in 1:nEpochs
            train_loss = 0.0; p̂1_loss = 0.0; p̂2_loss = 0.0; ŷ1_loss = 0.0; ŷ2_loss = 0.0; ŷ3_loss = 0.0;  ŷ4_loss = 0.0;  ŷ5_loss = 0.0; ŷ6_loss = 0.0
            batch_counter = 0
            for (x_batch, y_batch) in data_loader
                batch_counter += 1
                
                # Compute gradients
                grads, loss_batch, stats, train_st = Lux.Training.compute_gradients(vjp_rule, MCLoss, (x_batch, y_batch), train_st)

                # Update parameters
                Optimisers.update!(opt_st_backbone,     train_st.parameters[1],    grads[1]   )
                Optimisers.update!(opt_st_liqClassHead, train_st.parameters[2][1], grads[2][1])
                Optimisers.update!(opt_st_fluClassHead, train_st.parameters[2][2], grads[2][2])
                Optimisers.update!(opt_st_majOxHead,    train_st.parameters[2][3], grads[2][3])
                Optimisers.update!(opt_st_midOxHead,    train_st.parameters[2][4], grads[2][4])
                Optimisers.update!(opt_st_minOxHead,    train_st.parameters[2][5], grads[2][5])
                Optimisers.update!(opt_st_sys_Head,     train_st.parameters[2][6], grads[2][6])
                Optimisers.update!(opt_st_fra_Head,     train_st.parameters[2][7], grads[2][7])
                Optimisers.update!(opt_st_rho_Head,     train_st.parameters[2][8], grads[2][8])

                # Accumulate losses
                train_loss += loss_batch
                p̂1_loss  += stats[1]
                p̂2_loss  += stats[2]
                ŷ1_loss  += stats[3]
                ŷ2_loss  += stats[4]
                ŷ3_loss  += stats[5]
                ŷ4_loss  += stats[6]
                ŷ5_loss  += stats[7]
                ŷ6_loss  += stats[8]
            end

            # Adjust learning rates
            if epoch > warmup_epochs
                (η_bb,       T_bb      ) = adjust_learning_rate_cosine_annealing(T_bb,       T_tot; ηmin = 1e-3, ηmax = 1e-2)
                (η_liqClass, T_liqClass) = adjust_learning_rate_cosine_annealing(T_liqClass, T_tot; ηmin = 1e-3, ηmax = 1e-2)
                (η_fluClass, T_fluClass) = adjust_learning_rate_cosine_annealing(T_fluClass, T_tot; ηmin = 1e-3, ηmax = 1e-2)
                (η_mOx,      T_mOx     ) = adjust_learning_rate_cosine_annealing(T_mOx,      T_tot; ηmin = 1e-3, ηmax = 1e-2)
                (η_tOx,      T_tOx     ) = adjust_learning_rate_cosine_annealing(T_tOx,      T_tot; ηmin = 1e-3, ηmax = 1e-2)
                (η_sys,      T_sys     ) = adjust_learning_rate_cosine_annealing(T_sys,      T_tot; ηmin = 1e-3, ηmax = 1e-2)
                (η_liq,      T_liq     ) = adjust_learning_rate_cosine_annealing(T_liq,      T_tot; ηmin = 1e-3, ηmax = 1e-2)
                (η_flu,      T_flu     ) = adjust_learning_rate_cosine_annealing(T_flu,      T_tot; ηmin = 1e-3, ηmax = 5e-2)
                Optimisers.adjust!(opt_st_backbone,     η_bb      )
                Optimisers.adjust!(opt_st_liqClassHead, η_liqClass)
                Optimisers.adjust!(opt_st_fluClassHead, η_fluClass)
                Optimisers.adjust!(opt_st_majOxHead,    η_mOx     )
                Optimisers.adjust!(opt_st_midOxHead,    η_mOx     )
                Optimisers.adjust!(opt_st_minOxHead,    η_tOx     )
                Optimisers.adjust!(opt_st_sys_Head,      η_sys    )
                Optimisers.adjust!(opt_st_fra_Head,     η_liq     )
                Optimisers.adjust!(opt_st_rho_Head,     η_flu     )
            end

            # Early stopping
            es_counter, val_loss = early_stopping!(train_st, x_val_sc, cdev(y_val_sc), val_loss, es_counter)
            if isnan(train_loss) || isnan(val_loss) || isnan(p̂1_loss) || isnan(ŷ1_loss) || isnan(ŷ2_loss)
                @printf "NaN loss detected: epoch = %.5g \n train_loss = %.5g \t val_loss = %.5g \t p̂1_loss = %.5g \t p̂1_loss = %.5g \t ŷ1_loss = %.5g \t ŷ2_loss = %.5g \t ŷ3_loss = %.5g \t ŷ4_loss = %.5g \t ŷ5_loss = %.5g \t ŷ6_loss = %.5g\n" epoch (train_loss/batch_counter) val_loss (p̂1_loss/batch_counter) (p̂2_loss/batch_counter) (ŷ1_loss/batch_counter) (ŷ2_loss/batch_counter) (ŷ3_loss/batch_counter) (ŷ4_loss/batch_counter) (ŷ5_loss/batch_counter) (ŷ6_loss/batch_counter)
                error("NaNs in the losses")
            end

            # Make a prediction for the validation data
            test_st = Lux.testmode(train_st.states)
            (p̂1, p̂2, ŷ1, ŷ2, ŷ3, ŷ4, ŷ5, ŷ6) = cdev(Lux.apply(train_st.model.u, x_val_sc, train_st.parameters, test_st)[1])
            ŷ_sc = vcat(p̂1, vcat(p̂2, vcat(ŷ1, vcat(ŷ2, vcat(ŷ3, vcat(ŷ4, vcat(ŷ5, ŷ6)))))))
            TP_liq = mapreduce(i -> (p̂1[i] >= 0.5) & (cdev(y_val[1, i]) == 1.0), +, eachindex(p̂1))
            FP_liq = mapreduce(i -> (p̂1[i] >= 0.5) & (cdev(y_val[1, i]) == 0.0), +, eachindex(p̂1))
            FN_liq = mapreduce(i -> (p̂1[i] <  0.5) & (cdev(y_val[1, i]) == 1.0), +, eachindex(p̂1))
            TP_flu = mapreduce(i -> (p̂2[i] >= 0.5) & (cdev(y_val[2, i]) == 1.0), +, eachindex(p̂2))
            FP_flu = mapreduce(i -> (p̂2[i] >= 0.5) & (cdev(y_val[2, i]) == 0.0), +, eachindex(p̂2))
            FN_flu = mapreduce(i -> (p̂2[i] <  0.5) & (cdev(y_val[2, i]) == 1.0), +, eachindex(p̂2))
            precis_liq = TP_liq / (TP_liq + FP_liq) * 100.0
            recall_liq = TP_liq / (TP_liq + FN_liq) * 100.0
            precis_flu = TP_flu / (TP_flu + FP_flu) * 100.0
            recall_flu = TP_flu / (TP_flu + FN_flu) * 100.0
            ŷ = MAGEMin_MLPs.TransformDataBack(ŷ_sc; transform = :zscore, dims=2, mean_data = cdev(mean_y), std_data = cdev(std_y))

            # Revert logit transform
            if log_transform
                ŷ[[5, 7, 8, 9], :] .= exp.(ŷ[[5, 7, 8, 9], :]) .- DatType(ϵ_log)
            end

            # Early break condition
            if es_counter >= early_stop

                # Append losses and residuals for storage
                push!(train_loss_array, train_loss / batch_counter)
                push!(val_loss_array, val_loss)
                push!(p̂1_loss_array, p̂1_loss / batch_counter)
                push!(p̂2_loss_array, p̂2_loss / batch_counter)
                push!(ŷ1_loss_array, ŷ1_loss / batch_counter)
                push!(ŷ2_loss_array, ŷ2_loss / batch_counter)
                push!(ŷ3_loss_array, ŷ3_loss / batch_counter)
                push!(ŷ4_loss_array, ŷ4_loss / batch_counter)
                push!(ŷ5_loss_array, ŷ5_loss / batch_counter)
                push!(ŷ6_loss_array, ŷ6_loss / batch_counter)
                push!(precis_liq_array, precis_liq)
                push!(recall_liq_array, recall_liq)
                push!(precis_flu_array, precis_flu)
                push!(recall_flu_array, recall_flu)
                push!(R∑Xl_array, mean(sum(ŷ[3:10, :], dims = 1)  .- 1.0))
                overfit[k] = 1

                # Monitoring
                @printf "Epoch: %06d \t Train loss: %.3g \t Validation loss: %.3g \t p̂1: %.3g \t p̂2: %.3g \t ŷ1: %.3g \t ŷ2: %.3g\t ŷ3: %.3g \t ŷ4: %.3g \t ŷ5: %.3g \t ŷ6: %.3g \t precis_liq : %.3f \t recall_liq: %.3f \t precis_flu : %.3f \t recall_flu: %.3f\n" epoch (train_loss / batch_counter) val_loss (p̂1_loss / batch_counter) (p̂2_loss / batch_counter) (ŷ1_loss / batch_counter) (ŷ2_loss / batch_counter) (ŷ3_loss / batch_counter) (ŷ4_loss / batch_counter) (ŷ5_loss / batch_counter) (ŷ6_loss / batch_counter) precis_liq recall_liq precis_flu recall_flu
                @printf "Stopping early. No. bad consecutive epochs: %2d\n" es_counter
                @printf "Epoch : %d\n" epoch
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
                display([precis_liq_array[end], recall_liq_array[end]])
                display([precis_flu_array[end], recall_flu_array[end]])
                rel_error = mean(abs.(ŷ .- y_val) ./ max.(abs.(ŷ) .+ 1e-6, abs.(y_val) .+ 1e-6) .* 100.0, dims = 2)     # Add small value to avoid / 0
                @printf "Average relative prediction error\n"
                display(rel_error)
                @printf "R∑Xl : %.5g\n" R∑Xl_array[end]
                break
            end

            # Print to screen
            if epoch % nprint == 0 || epoch == nEpochs || epoch == 1
                # Append losses and residuals for storage
                push!(train_loss_array, train_loss / batch_counter)
                push!(val_loss_array, val_loss)
                push!(p̂1_loss_array, p̂1_loss / batch_counter)
                push!(p̂2_loss_array, p̂2_loss / batch_counter)
                push!(ŷ1_loss_array, ŷ1_loss / batch_counter)
                push!(ŷ2_loss_array, ŷ2_loss / batch_counter)
                push!(ŷ3_loss_array, ŷ3_loss / batch_counter)
                push!(ŷ4_loss_array, ŷ4_loss / batch_counter)
                push!(ŷ5_loss_array, ŷ5_loss / batch_counter)
                push!(ŷ6_loss_array, ŷ6_loss / batch_counter)
                push!(precis_liq_array, precis_liq)
                push!(recall_liq_array, recall_liq)
                push!(precis_flu_array, precis_flu)
                push!(recall_flu_array, recall_flu)
                push!(R∑Xl_array, mean(sum(ŷ[3:10, :], dims = 1)  .- 1.0))

                # Monitoring
                @printf "Epoch: %06d \t Train loss: %.3g \t Validation loss: %.3g \t p̂1: %.3g \t p̂2: %.3g \t ŷ1: %.3g \t ŷ2: %.3g\t ŷ3: %.3g \t ŷ4: %.3g \t ŷ5: %.3g \t ŷ6: %.3g \t precis_liq : %.3f \t recall_liq: %.3f \t precis_flu : %.3f \t recall_flu: %.3f\n" epoch (train_loss / batch_counter) val_loss (p̂1_loss / batch_counter) (p̂2_loss / batch_counter) (ŷ1_loss / batch_counter) (ŷ2_loss / batch_counter) (ŷ3_loss / batch_counter) (ŷ4_loss / batch_counter) (ŷ5_loss / batch_counter) (ŷ6_loss / batch_counter) precis_liq recall_liq precis_flu recall_flu

                # Last epoch - display error
                if epoch == nEpochs
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
                    display([precis_liq_array[end], recall_liq_array[end]])
                    display([precis_flu_array[end], recall_flu_array[end]])
                    rel_error = abs.(ŷ .- y_val) ./ max.(abs.(ŷ) .+ 1e-6, abs.(y_val) .+ 1e-6) .* 100.0    # Add small value to avoid / 0
                    @printf "Average relative prediction error\n"
                    display(mean(rel_error, dims = 2))
                    # idx = findall(rel_error[12, :] > 10.0)
                    # display(x_val[:, idx])
                    @printf "R∑Xl : %.5g\n" R∑Xl_array[end]
                end
            end
        end

        # Create structure
        trained_model = (
            StatefulLuxLayer{true}(mlp, cdev(train_st.parameters), cdev(train_st.states)),
            cdev(mean_x_tra),
            cdev(std_x_tra ),
            cdev(mean_y    ),
            cdev(std_y     ),
            cdev(x_tra     ),
            cdev(y_tra     ),
            cdev(x_val     ),
            cdev(y_val     ),
        )

        # Save model
        if save_model
            if k == 1
                isfile("$(fname_root)_crossvalMLP.jld2") ? run(`rm -rf $(fname_root)_crossvalMLP.jld2`) : nothing
                jldopen("$(fname_root)_crossvalMLP.jld2", "a") do file
                    info  = JLD2.Group(file, "Info")
                    info["nEpochs"   ] = nEpochs
                    info["nFolds"    ] = nFolds
                    info["nNeurons"  ] = hidden_dims
                    info["batchsize" ] = hyperparams["bsize"]
                end
            end
            jldopen("$(fname_root)_crossvalMLP.jld2", "a") do file
                fold  = JLD2.Group(file, "Fold_$(k)")
                MLP  = JLD2.Group(fold, "MLP")
                SCALE = JLD2.Group(fold, "Scaling")
                DATA  = JLD2.Group(fold, "Data")
                STATS = JLD2.Group(fold, "Stats")
                MLP["model"             ] = trained_model[1]
                SCALE["mean_input_train" ] = trained_model[2]
                SCALE["std_input_train"  ] = trained_model[3]
                SCALE["mean_output_train"] = trained_model[4]
                SCALE["std_output_train" ] = trained_model[5]
                DATA["input_train"       ] = trained_model[6]
                DATA["output_train"      ] = trained_model[7]
                DATA["input_val"         ] = trained_model[8]
                DATA["output_val"        ] = trained_model[9]
                STATS["train_losses"     ] = train_loss_array
                STATS["val_losses"       ] = val_loss_array
                STATS["p̂1_losses"        ] = p̂1_loss_array
                STATS["p̂2_losses"        ] = p̂2_loss_array
                STATS["ŷ1_losses"        ] = ŷ1_loss_array
                STATS["ŷ2_losses"        ] = ŷ2_loss_array
                STATS["ŷ3_losses"        ] = ŷ3_loss_array
                STATS["ŷ4_losses"        ] = ŷ4_loss_array
                STATS["ŷ5_losses"        ] = ŷ5_loss_array
                STATS["ŷ6_losses"        ] = ŷ6_loss_array
                STATS["precis_liq"       ] = precis_liq_array
                STATS["recall_liq"       ] = recall_liq_array
                STATS["precis_flu"       ] = precis_flu_array
                STATS["recall_flu"       ] = recall_flu_array
                STATS["R∑Xl"             ] = R∑Xl_array
                STATS["overfit"          ] = overfit
            end
        end

        # Return
        if k == nFolds
            return trained_model
        end

    end
end

"""
```
    train_model(x, y, dev; kwargs)
```

Trains the neural network with cross validation of the data set. Split between training and testing data is done internally.

# Arguments
* `x` : Input data
* `y` : Output data
* `dev` : Initialized device (CPU/GPU)
* `rng` : Random number generator

# Keyword arguments
* `nepochs` : No. of epochs to train
* `hidden_dims` : No. neurons per hidden layer
* `nprint` : Progress printing interval
* `save_model` : Save the trained model at each fold
* `fname_root` : Root for the file name. Used to store the trained model at each fold
* `log_transform` : Switch to activate log transformation
"""
@views function train_model(
    x             :: AbstractArray,
    y             :: AbstractArray,
    dev,
    rng;
    nEpochs       :: Int = 10_000,
    hidden_dims   :: Int = 32,
    hidden_lays   :: Int = 0,
    hyperparams   :: Dict{String, Real},
    nprint        :: Int = 250,
    save_model    :: Bool = false,
    fname_root    :: String,
    log_transform :: Bool = false,
    DatType       :: Type = Float32
)
    # Set CPU device
    cdev = cpu_device()

    # Set optimiser for model parts
    opt_backbone     = Optimisers.AdamW(1e-3, (0.9, 0.99), 1e-3, 1e-8)
    opt_liqClassHead = Optimisers.AdamW(1e-3, (0.9, 0.99), 1e-3, 1e-8)
    opt_fluClassHead = Optimisers.AdamW(1e-3, (0.9, 0.99), 1e-3, 1e-8)
    opt_majOxHead    = Optimisers.Adam(1e-3, (0.9, 0.99))
    opt_midOxHead    = Optimisers.Adam(1e-3, (0.9, 0.99))
    opt_minOxHead    = Optimisers.Adam(1e-3, (0.9, 0.99))
    opt_sys_Head     = Optimisers.Adam(1e-3, (0.9, 0.99))
    opt_fra_Head     = Optimisers.Adam(1e-3, (0.9, 0.99))
    opt_rho_Head     = Optimisers.Adam(1e-3, (0.9, 0.99))

    # Automatic differentiation
    vjp_rule = AutoZygote()

    # Hold out 1% for monitoring overfitting
    N                = size(x, 2)
    idx_tra, idx_val = MLUtils.kfolds(N, 100)

    # Define the MLP structure
    mlp = MLP(; hidden_dims = hidden_dims, hidden_lays = hidden_lays)

    # Lux setup
    if DatType == Float64
        mlp = Lux.f64(mlp)
    end
    params, state = Lux.setup(rng, mlp) |> dev

    if DatType == Float64
        params, state = Lux.f64(params), Lux.f64(state)
    end

    # Create training state
    train_st = Training.TrainState(mlp, params, state, opt_backbone)

    # Create optimiser states
    opt_st_backbone     = Optimisers.setup(opt_backbone,     params[1]   ) |> dev
    opt_st_liqClassHead = Optimisers.setup(opt_liqClassHead, params[2][1]) |> dev
    opt_st_fluClassHead = Optimisers.setup(opt_fluClassHead, params[2][2]) |> dev
    opt_st_majOxHead    = Optimisers.setup(opt_majOxHead,    params[2][3]) |> dev
    opt_st_midOxHead    = Optimisers.setup(opt_midOxHead,    params[2][4]) |> dev
    opt_st_minOxHead    = Optimisers.setup(opt_minOxHead,    params[2][5]) |> dev
    opt_st_sys_Head      = Optimisers.setup(opt_sys_Head,    params[2][6]) |> dev
    opt_st_fra_Head     = Optimisers.setup(opt_fra_Head,     params[2][7]) |> dev
    opt_st_rho_Head     = Optimisers.setup(opt_rho_Head,     params[2][8]) |> dev

    # Split data
    x_tra = deepcopy(x[:, idx_tra[1]]) |> x -> DatType.(x)
    y_tra = deepcopy(y[:, idx_tra[1]]) |> x -> DatType.(x)
    x_val = deepcopy(x[:, idx_val[1]]) |> x -> DatType.(x)
    y_val = deepcopy(y[:, idx_val[1]]) |> x -> DatType.(x)

    @show intersect(eachcol(y_val), eachcol(y_tra))

    # Logit transform
    if log_transform
        x_tra[[2, 5, 6, 8, 9], :] .= log.(x_tra[[2, 5, 6, 8, 9], :] .+ DatType(ϵ_log))
        y_tra[[5, 7, 8, 9],    :] .= log.(y_tra[[5, 7, 8, 9],    :] .+ DatType(ϵ_log))
        x_val[[2, 5, 6, 8, 9], :] .= log.(x_val[[2, 5, 6, 8, 9], :] .+ DatType(ϵ_log))
        y_val[[5, 7, 8, 9],    :] .= log.(y_val[[5, 7, 8, 9],    :] .+ DatType(ϵ_log))
    end

    # Scale training data (only continuous part)
    mask_liq = y_tra[1, :] .> 0.0
    mask_flu = y_tra[2, :] .> 0.0
    mean_y   = zeros(DatType, size(y_tra, 1), 1)
    std_y    = zeros(DatType, size(y_tra, 1), 1)
    x_tra_sc, mean_x_tra, std_x_tra = MAGEMin_MLPs.TransformData(x_tra, transform = :zscore, dims = 2; DatType = DatType)
    _, mean_y_Xl, std_y_Xl          = MAGEMin_MLPs.TransformData(y_tra[3:10, mask_liq], transform = :zscore, dims = 2; DatType = DatType)
    _, mean_y_ρS, std_y_ρS          = MAGEMin_MLPs.TransformData(y_tra[11, :], transform = :zscore, dims = 1; DatType = DatType)
    _, mean_y_l,  std_y_l           = MAGEMin_MLPs.TransformData(y_tra[[12, 14], mask_liq], transform = :zscore, dims = 2; DatType = DatType)
    _, mean_y_f,  std_y_f           = MAGEMin_MLPs.TransformData(y_tra[[13, 15], mask_flu], transform = :zscore, dims = 2; DatType = DatType)

    # Add unscaled outputs
    mean_y[[1, 2],    1] .= 0.0
    mean_y[3:10,      1] .= mean_y_Xl
    mean_y[11,        1]  = mean_y_ρS[1]
    mean_y[[12, 14],  1] .= mean_y_l
    mean_y[[13, 15],  1] .= mean_y_f
    std_y[[1, 2],     1] .= 1.0
    std_y[3:10,       1] .= std_y_Xl
    std_y[11,         1]  = std_y_ρS[1]
    std_y[[12, 14],   1] .= std_y_l
    std_y[[13, 15],   1] .= std_y_f
    y_tra_sc = MAGEMin_MLPs.TransformData(y_tra, transform = :zscore, scale = true, dims = 2, mean_data = mean_y, std_data = std_y; DatType = DatType)
    dev(mean_y)
    dev(std_y)

    # Scale validation data to the mean and standard deviation of the training set
    x_val_sc = MAGEMin_MLPs.TransformData(x_val, transform = :zscore, dims = 2, scale = true, mean_data = cdev(mean_x_tra), std_data = cdev(std_x_tra), DatType = DatType) |> dev
    y_val_sc = MAGEMin_MLPs.TransformData(y_val, transform = :zscore, dims = 2, scale = true, mean_data = cdev(mean_y    ), std_data = cdev(std_y    ), DatType = DatType) |> dev

    # Revert logit transform for validation set
    if log_transform
            y_val[[5, 7, 8, 9], :] .= exp.(y_val[[5, 7, 8, 9], :]) .- DatType(ϵ_log)
    end

    # Create data loader
    data_loader = DataLoader((x_tra_sc, y_tra_sc); batchsize = hyperparams["bsize"], shuffle = false, rng = rng) |> dev

    # Initialize losses and counters
    es_counter = 0
    train_loss_array = []; p̂1_loss_array = [] ; p̂2_loss_array = [] ; ŷ1_loss_array = []; ŷ2_loss_array = []; ŷ3_loss_array = []; ŷ4_loss_array = []; ŷ5_loss_array = []; ŷ6_loss_array = []

    # Training loop
    val_loss = 10_000
    warmup_epochs = 50
    T_bb = 1; T_liqClass = 1; T_fluClass = 1; T_mOx = 1; T_tOx = 1; T_liq = 1; T_sys = 1; T_flu = 1; T_tot = 250
    η_bb = 0.0; η_liqClass = 0.0; η_fluClass = 0.0; η_mOx = 0.0; η_tOx = 0.0; η_liq = 0.0; η_sys = 0.0; η_flu = 0.0
    @printf "Starting training ...\n"
    for epoch in 1:nEpochs
        train_loss = 0.0; p̂1_loss = 0.0; p̂2_loss = 0.0; ŷ1_loss = 0.0; ŷ2_loss = 0.0; ŷ3_loss = 0.0;  ŷ4_loss = 0.0;  ŷ5_loss = 0.0; ;  ŷ6_loss = 0.0
        batch_counter = 0
        for (x_batch, y_batch) in data_loader
            batch_counter += 1

            # Compute gradients
            grads, loss_batch, stats, train_st = Lux.Training.compute_gradients(vjp_rule, MCLoss, (x_batch, y_batch), train_st)

            # Update parameters
            Optimisers.update!(opt_st_backbone,     train_st.parameters[1],    grads[1]   )
            Optimisers.update!(opt_st_liqClassHead, train_st.parameters[2][1], grads[2][1])
            Optimisers.update!(opt_st_fluClassHead, train_st.parameters[2][2], grads[2][2])
            Optimisers.update!(opt_st_majOxHead,    train_st.parameters[2][3], grads[2][3])
            Optimisers.update!(opt_st_midOxHead,    train_st.parameters[2][4], grads[2][4])
            Optimisers.update!(opt_st_minOxHead,    train_st.parameters[2][5], grads[2][5])
            Optimisers.update!(opt_st_sys_Head,     train_st.parameters[2][6], grads[2][6])
            Optimisers.update!(opt_st_fra_Head,     train_st.parameters[2][7], grads[2][7])
            Optimisers.update!(opt_st_rho_Head,     train_st.parameters[2][8], grads[2][8])

            # Accumulate losses
            train_loss += loss_batch
            p̂1_loss  += stats[1]
            p̂2_loss  += stats[2]
            ŷ1_loss  += stats[3]
            ŷ2_loss  += stats[4]
            ŷ3_loss  += stats[5]
            ŷ4_loss  += stats[6]
            ŷ5_loss  += stats[7]
            ŷ6_loss  += stats[8]
        end

        # Adjust learning rates
        if epoch > warmup_epochs
            (η_bb,       T_bb      ) = adjust_learning_rate_cosine_annealing(T_bb,       T_tot; ηmin = 1e-3, ηmax = 1e-2)
            (η_liqClass, T_liqClass) = adjust_learning_rate_cosine_annealing(T_liqClass, T_tot; ηmin = 1e-3, ηmax = 1e-2)
            (η_fluClass, T_fluClass) = adjust_learning_rate_cosine_annealing(T_fluClass, T_tot; ηmin = 1e-3, ηmax = 1e-2)
            (η_mOx,      T_mOx     ) = adjust_learning_rate_cosine_annealing(T_mOx,      T_tot; ηmin = 1e-3, ηmax = 1e-2)
            (η_tOx,      T_tOx     ) = adjust_learning_rate_cosine_annealing(T_tOx,      T_tot; ηmin = 1e-3, ηmax = 1e-2)
            (η_sys,      T_sys     ) = adjust_learning_rate_cosine_annealing(T_sys,      T_tot; ηmin = 1e-3, ηmax = 1e-2)
            (η_liq,      T_liq     ) = adjust_learning_rate_cosine_annealing(T_liq,      T_tot; ηmin = 1e-3, ηmax = 1e-2)
            (η_flu,      T_flu     ) = adjust_learning_rate_cosine_annealing(T_flu,      T_tot; ηmin = 1e-3, ηmax = 5e-2)
            Optimisers.adjust!(opt_st_backbone,     η_bb      )
            Optimisers.adjust!(opt_st_liqClassHead, η_liqClass)
            Optimisers.adjust!(opt_st_fluClassHead, η_fluClass)
            Optimisers.adjust!(opt_st_majOxHead,    η_mOx     )
            Optimisers.adjust!(opt_st_midOxHead,    η_mOx     )
            Optimisers.adjust!(opt_st_minOxHead,    η_tOx     )
            Optimisers.adjust!(opt_st_sys_Head,      η_sys    )
            Optimisers.adjust!(opt_st_fra_Head,     η_liq     )
            Optimisers.adjust!(opt_st_rho_Head,     η_flu     )
        end

        # Early stopping
        es_counter, val_loss = early_stopping!(train_st, x_val_sc, cdev(y_val_sc), val_loss, es_counter)
        if isnan(train_loss) || isnan(val_loss) || isnan(p̂1_loss) || isnan(ŷ1_loss) || isnan(ŷ2_loss)
            @printf "NaN loss detected: epoch = %.5g \n train_loss = %.5g \t val_loss = %.5g \t p̂1_loss = %.5g \t p̂1_loss = %.5g \t ŷ1_loss = %.5g \t ŷ2_loss = %.5g \t ŷ3_loss = %.5g \t ŷ4_loss = %.5g \t ŷ5_loss = %.5g \t ŷ6_loss = %.5g\n" epoch (train_loss/batch_counter) val_loss (p̂1_loss/batch_counter) (p̂2_loss/batch_counter) (ŷ1_loss/batch_counter) (ŷ2_loss/batch_counter) (ŷ3_loss/batch_counter) (ŷ4_loss/batch_counter) (ŷ5_loss/batch_counter) (ŷ6_loss/batch_counter)
            error("NaNs in the losses")
        end

        # Print to screen
        if epoch % nprint == 0 || epoch == nEpochs || epoch == 1 || es_counter >= 10
            # Append losses and residuals for storage
            push!(train_loss_array, train_loss / batch_counter)
            push!(p̂1_loss_array, p̂1_loss / batch_counter)
            push!(p̂2_loss_array, p̂2_loss / batch_counter)
            push!(ŷ1_loss_array, ŷ1_loss / batch_counter)
            push!(ŷ2_loss_array, ŷ2_loss / batch_counter)
            push!(ŷ3_loss_array, ŷ3_loss / batch_counter)
            push!(ŷ4_loss_array, ŷ4_loss / batch_counter)
            push!(ŷ5_loss_array, ŷ5_loss / batch_counter)
            push!(ŷ6_loss_array, ŷ6_loss / batch_counter)

            @printf "Epoch: %06d \t Train loss: %.3g \t p̂1: %.3g \t p̂2: %.3g \t ŷ1: %.3g \t ŷ2: %.3g\t ŷ3: %.3g \t ŷ4: %.3g \t ŷ5: %.3g \t ŷ6: %.3g \n" epoch (train_loss / batch_counter) (p̂1_loss / batch_counter) (p̂2_loss / batch_counter) (ŷ1_loss / batch_counter) (ŷ2_loss / batch_counter) (ŷ3_loss / batch_counter) (ŷ4_loss / batch_counter) (ŷ5_loss / batch_counter) (ŷ6_loss / batch_counter)
        end

        # Stop early to prevent overfitting
        if es_counter >= 10
            break
        end
    end

    # Create structure
    trained_model = (
        StatefulLuxLayer{true}(mlp, cdev(train_st.parameters), cdev(train_st.states)),
        cdev(mean_x_tra),
        cdev(std_x_tra ),
        cdev(mean_y),
        cdev(std_y ),
        cdev(x_tra),
        cdev(y_tra),
    )

    # Save model
    if save_model
        isfile("$(fname_root)_full.jld2") ? run(`rm -rf $(fname_root)_full.jld2`) : nothing
        jldopen("$(fname_root)_full.jld2", "a") do file
            info  = JLD2.Group(file, "Info")
            MLP   = JLD2.Group(file, "MLP")
            SCALE = JLD2.Group(file, "Scaling")
            DATA  = JLD2.Group(file, "Data")
            STATS = JLD2.Group(file, "Stats")
            info["nEpochs"          ] = nEpochs
            info["nNeurons"         ] = hidden_dims
            info["batchsize"        ] = hyperparams["bsize"]
            MLP["model"             ] = trained_model[1]
            SCALE["mean_input"      ] = trained_model[2]
            SCALE["std_input"       ] = trained_model[3]
            SCALE["mean_output"     ] = trained_model[4]
            SCALE["std_output"      ] = trained_model[5]
            DATA["input"            ] = trained_model[6]
            DATA["output"           ] = trained_model[7]
            STATS["train_losses"    ] = train_loss_array
            STATS["fluidClass_loss" ] = p̂1_loss_array
            STATS["liquidClass_loss"] = p̂2_loss_array
            STATS["meltmajOx_loss"  ] = ŷ1_loss_array
            STATS["meltmidOx_loss"  ] = ŷ2_loss_array
            STATS["meltminOx_loss"  ] = ŷ3_loss_array
            STATS["sys_loss"    ] = ŷ4_loss_array
            STATS["fra_loss"] = ŷ5_loss_array
            STATS["rho_loss" ] = ŷ6_loss_array
        end
    end

    # Return
    return trained_model
end

"""
```
    MCLoss(model, params, state, data)
```
Custom defined loss function. The `data` tuple contains the training input and output data.

# Arguments
* `model` : Model structure provided by `Lux.jl`
* `params` : Parameters of `Lux` model
* `state` : State of `Lux` model
* `data` : Tuple containg all relevant data
"""
@views function MCLoss(
    model,
    params,
    state,
    (x, y)
)
    
    # Make a prediction
    states_t  = Lux.trainmode(state)
    net       = StatefulLuxLayer{true}(model.u, params, states_t)
    (p̂1, p̂2, ŷ1, ŷ2, ŷ3, ŷ4, ŷ5, ŷ6)  = net(x)

    # Individual losses
    p̂1_loss  = Lux.BinaryCrossEntropyLoss(; epsilon = 1e-8)(p̂1', y[1, :])
    p̂2_loss  = Lux.BinaryCrossEntropyLoss(; epsilon = 1e-8)(p̂2', y[2, :])
    ŷ1a_loss = Lux.HuberLoss()((y[1,:] .== 1) .* ŷ1[1, :], (y[1,:] .== 1) .* y[3,  :] )
    ŷ1b_loss = Lux.HuberLoss()((y[1,:] .== 1) .* ŷ1[2, :], (y[1,:] .== 1) .* y[4,  :] )
    ŷ2a_loss = Lux.HuberLoss()((y[1,:] .== 1) .* ŷ2[1, :], (y[1,:] .== 1) .* y[5,  :] )
    ŷ2b_loss = Lux.HuberLoss()((y[1,:] .== 1) .* ŷ2[2, :], (y[1,:] .== 1) .* y[6,  :] )
    ŷ2c_loss = Lux.HuberLoss()((y[1,:] .== 1) .* ŷ2[3, :], (y[1,:] .== 1) .* y[7,  :] )
    ŷ3a_loss = Lux.HuberLoss()((y[1,:] .== 1) .* ŷ3[1, :], (y[1,:] .== 1) .* y[8,  :] )
    ŷ3b_loss = Lux.HuberLoss()((y[1,:] .== 1) .* ŷ3[2, :], (y[1,:] .== 1) .* y[9,  :] )
    ŷ3c_loss = Lux.HuberLoss()((y[1,:] .== 1) .* ŷ3[3, :], (y[1,:] .== 1) .* y[10, :] )
    ŷ4_loss  = Lux.HuberLoss()(                  ŷ4,                         y[11, :]')
    ŷ5a_loss = Lux.HuberLoss()((y[1,:] .== 1) .* ŷ5[1, :], (y[1,:] .== 1) .* y[12, :] )
    ŷ5b_loss = Lux.HuberLoss()((y[2,:] .== 1) .* ŷ5[2, :], (y[2,:] .== 1) .* y[13, :] )
    ŷ6a_loss = Lux.HuberLoss()((y[1,:] .== 1) .* ŷ6[1, :], (y[1,:] .== 1) .* y[14, :] )
    ŷ6b_loss = Lux.HuberLoss()((y[2,:] .== 1) .* ŷ6[2, :], (y[2,:] .== 1) .* y[15, :] )

    # Sum up losses
    loss = p̂1_loss + p̂2_loss + (ŷ1a_loss + ŷ1b_loss) + (ŷ2a_loss + ŷ2b_loss + ŷ2c_loss) + (ŷ3a_loss + ŷ3b_loss + ŷ3c_loss) + ŷ4_loss + (ŷ5a_loss + ŷ5b_loss) + (ŷ6a_loss + ŷ6b_loss)

    # Return
    return (loss, state, (p̂1_loss, p̂2_loss, (ŷ1a_loss + ŷ1b_loss), (ŷ2a_loss + ŷ2b_loss + ŷ2c_loss), (ŷ3a_loss + ŷ3b_loss + ŷ3c_loss), ŷ4_loss, (ŷ5a_loss + ŷ5b_loss), (ŷ6a_loss + ŷ6b_loss)))
end

"""
```
    adjust_learning_rate_cosine_annealing(T_cur, T_freq; ...kwargs)
```

Adjusts the learning rate according to the current epoch.

# Arguments
* `T_cur` : No. of current epoch w.r.t. last reset
* `T_freq` : Frequency with which the learning rate should be reset

# Keyword arguments:
* `ηmin` : Minimum bound of learning rate
* `ηmax` : Maximum bound of learning rate
"""
function adjust_learning_rate_cosine_annealing(
    T_cur   :: Int64,
    T_freq  :: Int64;
    ηmin    :: Float64 = 0.01,
    ηmax    :: Float64 = 0.1
)
    T_cur == T_freq + 1 ? T_cur = 0 : nothing
    return (ηmin + 0.5 * (ηmax - ηmin) * (1.0 + cos(T_cur / T_freq * π)), T_cur + 1)
end

"""
```
    SplitData(data, rng; ...kwargs)
````

Returns the random indices of two sets. The split is performed according to a given percentage.

# Arguments:
* `data` : `AbstractArray` containing the data to split
* `rng`  : Random number generator

# Keyword arguments:
* nPerc : Percentage for the split
"""
@views function SplitData(
    data   :: AbstractArray,
    rng;
    nPerc :: Float64 = 20.0
)
    # Define variables
    Npts      = size(data, 2)               # Data size
    N         = Int64(ceil(size(data, 2) * nPerc / 100))   # Absolute testing point no.
    idx_feat  = 10                           # Feature index to sample from
    
    # Initialize arrays
    idx_test = Int64[]
    idx_train = collect(1:Npts)

    # Identifiy testing points
    idx_test = rand(rng, idx_train, N) |> idx -> unique(idx) |> idx -> sort(idx)
    deleteat!(idx_train, sort(idx_test))

    # Return a named tuple containing the required indices
    return (idx_train=idx_train, idx_test=idx_test)
end

function load_igneous_model(
    mfile :: String
)
    # Load model
    model  = JLD2.load(mfile, "/MLP/model")
    x_mean = JLD2.load(mfile, "Scaling/mean_input")
    x_std  = JLD2.load(mfile, "Scaling/std_input")
    y_mean = JLD2.load(mfile, "Scaling/mean_output")
    y_std  = JLD2.load(mfile, "Scaling/std_output")

    # Create output structure
    ig_NN = (model=model, in_mean=x_mean, in_std=x_std, out_mean=y_mean, out_std=y_std)

    # Return structure
    return ig_NN
end

function infer_igneous_properties(
    ig_NN  :: NamedTuple,
    T      :: Float64,
    P      :: Float64,
    X      :: Vector{Float64},
    oxides :: Vector{String}
)
    # Get no. points and set network expected order of bulk oxide input
    npts = size(T, 1)
    Noxides = ["SiO2", "Al2O3", "CaO", "MgO", "FeO", "K2O+Na2O", "TiO2", "H2O"]

    # Assemble input matrix and apply transformations
    input = Matrix{Float32}(undef, 10, npts)
    input[1, :] .= Float32.(T)
    input[2, :] .= Float32.(P)
    for (idx_ox, oxide) in enumerate(Noxides)
        idx = findfirst(x -> x == oxide, oxides)
        input[2+idx_ox, :] .= Float32.(X[idx, :])
    end
    sum_X = sum(input[3:10, :], dims = 1)
    input[3:10, :] ./= sum_X
    input[[2, 5, 6, 8, 9], :] .= log.(input[[2, 5, 6, 8, 9], :] .+ Float32(1e-5))
    input_sc = MAGEMin_MLPs.TransformData(input, transform = :zscore, dims = 2, scale = true, mean_data = ig_NN.in_mean, std_data = ig_NN.in_std)
    
    # Infer properties
    states_t = Lux.testmode(ig_NN.model.st)
    (p̂1, p̂2, ŷ1, ŷ2, ŷ3, ŷ4, ŷ5, ŷ6)  = Lux.apply(ig_NN.model.model.u, input_sc, ig_NN.model.ps, states_t)[1]
    ŷ_sc = vcat(p̂1, vcat(p̂2, vcat(ŷ1, vcat(ŷ2, vcat(ŷ3, vcat(ŷ4, vcat(ŷ5, ŷ6)))))))
    ŷ  = MAGEMin_MLPs.TransformDataBack(ŷ_sc; transform = :zscore, dims=2, mean_data = ig_NN.out_mean, std_data = ig_NN.out_std)
    ŷ[[5, 7, 8, 9], :] .= exp.(ŷ[[5, 7, 8, 9], :]) .- Float32(1e-5)
    ŷ[3:10]     ./= sum(ŷ[3:10])


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
    ŷ = abs.(ŷ)
    
    # Correct points that are close to the aggregate transitions
    thres = 5e-2
    ph_mob      = vec(sum(ŷ[[12, 13], :], dims = 1))
    idx_solid   = findall(x -> x < 1e-3,       ph_mob)
    idx_molten  = findall(x -> x > 1.0 - thres, ph_mob)
    ŷ[[12, 13, 14, 15], idx_solid       ] .= 0.0
    idx_dens_melt = ŷ[14, :] .> ŷ[11, :]
    ŷ[14, idx_dens_melt] .= ŷ[11, idx_dens_melt]
    ρ_sol = (ŷ[11, :] .- ŷ[12, :] .* ŷ[14, :] .- ŷ[13, :] .* ŷ[15, :]) ./ (1.0 .- ŷ[12, :] .- ŷ[13, :])
    ρ_sol[idx_molten] .= 0.0

    # Return
    return (X_liq = Float64.(ŷ[3:10, :]), ρ_sys = Float64.(ŷ[11, :]), ϕ_liq = Float64.(ŷ[12, :]), ϕ_flu = Float64.(ŷ[13, :]), ρ_liq = Float64.(ŷ[14, :]), ρ_flu = Float64.(ŷ[15, :]))
end