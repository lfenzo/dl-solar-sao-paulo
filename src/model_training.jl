using MLJ
using CSV
using Flux
using CUDA
using BSON
using Dates
using FileIO
using DataFrames
using CairoMakie
using ProgressBars
using BenchmarkTools


const DATA_DIR = joinpath(pwd(), "..", "data")
const ARTIFACT_DIR = joinpath(pwd(), "..", "artifacts")


function train(; xtrain, ytrain, xvalid, yvalid, epochs::Integer, force_cpu::Bool) 

    loss_train = Vector{Float32}()
    loss_valid = Vector{Float32}()

    if force_cpu || !CUDA.functional()
        @info "Training on CPU"
        device = cpu
    else
        @info "Training on GPU"
        CUDA.reclaim()
        device = gpu
    end

    train_loader = Flux.DataLoader((xtrain, ytrain) |> device, batchsize = 128, shuffle = true)
    valid_loader = Flux.DataLoader((xvalid, yvalid) |> device, batchsize = 128, shuffle = true)

    model = Chain(
        Dense(size(xtrain, 1) => 330, relu),
        Dropout(0.4),
        Dense(330 => 330, relu),
        Dropout(0.4),
        Dense(330 => 50, relu),
        Dropout(0.3),
        Dense(50 => 1, identity),
    ) |> device

    optimizer = Flux.Adam(0.0002)
    parameters = Flux.params(model)

    for epoch in 1:epochs
        for ((xtr, ytr), (xval, yval)) in ProgressBar(zip(train_loader, valid_loader))
            grads = gradient(() -> Flux.Losses.mse(model(xtr), ytr), parameters)
            Flux.Optimise.update!(optimizer, parameters, grads)
        end

        epoch_train_loss = Flux.Losses.mse(model(xtrain), ytrain)
        epoch_valid_loss = Flux.Losses.mse(model(xvalid), yvalid)

        @info "Epoch $epoch\n" *
            "training loss \t$epoch_train_loss\n" *
            "validation loss \t$epoch_valid_loss"

        push!(loss_train, epoch_train_loss)
        push!(loss_valid, epoch_valid_loss)

        plot_loss_curves(loss_train, loss_valid)
    end

    return cpu(model), loss_train, loss_valid
end


function evaluate(model, x, y; output_scaler)

    comparison = copy(y)

    comparison.norm_predicted = model(Array(x)')[1, :]
    comparison.predicted = inverse_transform(output_scaler, comparison.norm_predicted)

    @info first(comparison, 12)
    @info "MAE (rescaled) = $(Flux.Losses.mae(comparison.predicted, comparison.next_radiation))"
    @info "MAE (normalized) = $(Flux.Losses.mae(comparison.norm_predicted, comparison.norm_next_radiation))"
end


function plot_loss_curves(loss_train::Vector, loss_valid::Vector)
    fig = Figure()
    axs = Axis(fig[1, 1])
    
    lines!(axs, loss_train, label = "Train")
    lines!(axs, loss_valid, label = "Valid")

    axislegend()
    axs.xlabel = "Epochs"
    axs.ylabel = "Loss"

    save("learning-curves-train-valid.pdf", fig)
end


function main()

    #
    # Loading training and validation data
    #

    @info "Loading validation and training data"
    xtrain, ytrain = FileIO.load(joinpath(ARTIFACT_DIR, "artifacts.jld2"), "xtrain", "ytrain")
    xvalid, yvalid = FileIO.load(joinpath(ARTIFACT_DIR, "artifacts.jld2"), "xvalid", "yvalid")

    xtrain = first(xtrain, 100_000)
    ytrain = first(ytrain, 100_000)

    #
    # Loading the scaler objects
    #

    @info "Loading scaler objects for input and output data"
    data_transformer = FileIO.load(joinpath(ARTIFACT_DIR, "artifacts.jld2"), "input_scaler") 
    target_transformer = FileIO.load(joinpath(ARTIFACT_DIR, "artifacts.jld2"), "output_scaler") 

    #
    # Training the model
    #

    @info "Training with $(size(xtrain, 1)) samples"

    epochs = 40
    start_datestamp = Dates.now()

    # removing the extra features for the Y sets (NOTE that the target feature is
    # garanteed to be the last feature, this is set in the data pipeline script)
    @time model, loss_train, loss_valid = train(
        xtrain = Array(xtrain)',
        ytrain = Array(ytrain[:, :norm_next_radiation])',
        xvalid = Array(xvalid)',
        yvalid = Array(yvalid[:, :norm_next_radiation])',
        epochs = epochs,
        force_cpu = true,
    )

    #
    # Preliminary model evaluation
    #

    evaluate(model, xvalid, yvalid; output_scaler = target_transformer,)
end


main()
