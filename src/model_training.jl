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
using ParameterSchedulers


const ARTIFACT_DIR = joinpath(pwd(), "..", "artifacts")
const MODEL_DIR = joinpath(pwd(), "..", "models")
const PLOT_DIR = joinpath(ARTIFACT_DIR, "learning_curves")


function build_model(; input_size::Integer, output_size::Integer) :: Chain
    return Chain(

        Dense(input_size => 400, selu),
        Dropout(0.30),
        BatchNorm(400),

        Dense(400 => 300, selu),
        Dropout(0.30),
        BatchNorm(300),

        Dense(300 => 200, selu),
        Dropout(0.30),
        BatchNorm(200),

        Dense(200 => output_size, identity),
    )
end


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

    batchsize = 64

    train_loader = Flux.DataLoader((xtrain, ytrain) |> device; batchsize = batchsize, shuffle = true)
    valid_loader = Flux.DataLoader((xvalid, yvalid) |> device; batchsize = batchsize, shuffle = true)

    model = build_model(
        input_size = size(xtrain, 1),
        output_size = size(ytrain, 1)
    ) |> device

    parameters = Flux.params(model)
    optimizer = Adam()
    learning_rate_scheduler = ParameterSchedulers.Exp(
        λ = 0.003,
        γ = 0.95,
    )

    for (epoch, lr) in zip(1:epochs, learning_rate_scheduler)

        optimizer.eta = lr  # updating the llearning rate based on the scheduler

        for ((xtr, ytr), (xval, yval)) in ProgressBar(zip(train_loader, valid_loader))
            grads = gradient(() -> Flux.Losses.mse(model(xtr), ytr), parameters)
            Flux.Optimise.update!(optimizer, parameters, grads)
        end

        epoch_train_loss = Flux.Losses.mse(model(xtrain), ytrain)
        epoch_valid_loss = Flux.Losses.mse(model(xvalid), yvalid)

        @info "Epoch $epoch | Learning Rate: $(optimizer.eta)\n" *
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


function plot_loss_curves(loss_train::Vector, loss_valid::Vector, model = nothing)
    fig = Figure()
    axs = Axis(
        fig[1, 1];
        xlabel = "Epochs",
        ylabel = "Loss",
    )
    
    lines!(axs, loss_train, label = "Train")
    lines!(axs, loss_valid, label = "Valid")
    axislegend()

    if model !== nothing
        save(joinpath(PLOT_DIR, "$model.pdf"), fig)
    else
        save(joinpath(ARTIFACT_DIR, "learning_curves_last_model.pdf"), fig)
    end

    return fig
end


function main()

    !isdir(PLOT_DIR) && mkdir(PLOT_DIR)
    !isdir(MODEL_DIR) && mkdir(MODEL_DIR)

    #
    # Loading training and validation data
    #

    @info "Loading validation and training data"
    xtrain, ytrain = FileIO.load(joinpath(ARTIFACT_DIR, "artifacts.jld2"), "xtrain", "ytrain")
    xvalid, yvalid = FileIO.load(joinpath(ARTIFACT_DIR, "artifacts.jld2"), "xvalid", "yvalid")
    xtest, ytest = FileIO.load(joinpath(ARTIFACT_DIR, "artifacts.jld2"), "xtest", "ytest")

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
    epochs = 20

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

    end_datestamp = Dates.format(Dates.now(), "Y-m-d-H-M")
    model_name = "model_$end_datestamp"

    @info "Saving the model $(model_name).bson"
    BSON.@save(joinpath(MODEL_DIR, "$(model_name).bson"), model)

    @info "Saving the model learning curves in $(model_name).pdf"
    plot_loss_curves(loss_train, loss_valid, "$(model_name).pdf")

    #
    # Preliminary model evaluation
    #

    evaluate(model, xtest, ytest; output_scaler = target_transformer)
end


main()
