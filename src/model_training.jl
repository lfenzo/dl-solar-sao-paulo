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


"""
    build_model(; input_size::Integer, output_size::Integer) :: Chain

Build a sequential Dense Neural Network having `input_size` input neurons and `output_size` output
neurons. 
"""
function build_model(; input_size::Integer, output_size::Integer) :: Chain
    return Chain(

        Dense(input_size => 400, relu),
        Dropout(0.30),
        BatchNorm(400),

        Dense(400 => 400, relu),
        Dropout(0.30),
        BatchNorm(400),

        Dense(400 => 300, relu),
        Dropout(0.30),
        BatchNorm(300),

        Dense(300 => 300, relu),
        Dropout(0.30),
        BatchNorm(300),

        Dense(300 => 200, relu),
        Dropout(0.30),
        BatchNorm(200),

        Dense(200 => output_size, identity),
    )
end


function loss_metrics(model::Chain, loader, device)
    mse = 0.0
    m = 0
    for (x, y) in loader
        x, y = device(x), device(y)
        mse += Flux.Losses.mse(model(x), y) 
        m += size(x)[end]
    end
    return mse / m
end


"""
    train(; xtrain, ytrain, xvalid, yvalid, epochs::Integer, force_cpu::Bool)

Train a neural network with the dat in `xtrain` and `ytrain` for a total of `epochs` epochs. During
the fitting process both `xvalid` and `yvalid` are used to assess the quality of convergence.
Optinally use `use_gpu = true` to force the training in the GPU (if available).

# Returns

A tuple with 3 elements:
1. the trained model
2. a vector with per-epoch values for the training data loss
3. a vector with per-epoch values for the validation data loss
"""
function train(; xtrain, ytrain, xvalid, yvalid, epochs::Integer, use_gpu::Bool) 

    loss_train = Vector{Float32}()
    loss_valid = Vector{Float32}()

    if use_gpu && CUDA.functional()
        CUDA.reclaim()
        device = gpu
    else
        device = cpu
    end

    batchsize = 256

    train_loader = Flux.DataLoader((xtrain, ytrain), batchsize = batchsize, shuffle = true)
    valid_loader = Flux.DataLoader((xvalid, yvalid), batchsize = batchsize)

    model = build_model(
        input_size = size(xtrain, 1),
        output_size = size(ytrain, 1)
    ) |> device

    optimizer = Adam()
    parameters = Flux.params(model)
    learning_rate_scheduler = ParameterSchedulers.Exp(
        λ = 0.003,
        γ = 0.95,
    )

    for (epoch, lr) in zip(1:epochs, learning_rate_scheduler)

        optimizer.eta = lr  # updating the llearning rate based on the scheduler

        for (xtr, ytr) in ProgressBar(train_loader)
            x, y = device(xtr), device(ytr)
            grads = gradient(() -> Flux.Losses.mse(model(x), y), parameters)
            Flux.Optimise.update!(optimizer, parameters, grads)
        end

        epoch_train_loss = loss_metrics(model, train_loader, device)
        epoch_valid_loss = loss_metrics(model, valid_loader, device)

        @info "Epoch $epoch | Learning Rate: $(optimizer.eta)\n" *
            "training loss \t$epoch_train_loss\n" *
            "validation loss \t$epoch_valid_loss"

        push!(loss_train, epoch_train_loss)
        push!(loss_valid, epoch_valid_loss)

        plot_loss_curves(loss_train, loss_valid)
    end

    return cpu(model), loss_train, loss_valid
end


"""
    evaluate(model, x, y; output_scaler)

Evaluate some model `model` using the input data in `x` by comparing its output to `y`. Since
the de-normalization operation has to be carried out, a `output_scaler` must also be specified.
"""
function evaluate(model, x, y; output_scaler)
    comparison = copy(y)
    comparison.norm_predicted = model(Array(x)')[1, :]
    comparison.predicted = inverse_transform(output_scaler, comparison.norm_predicted)

    @info first(comparison, 12)
    @info "MAE (rescaled) = " *
        string(Flux.Losses.mae(comparison.predicted, comparison.next_radiation))
    @info "MAE (normalized) = " *
        string(Flux.Losses.mae(comparison.norm_predicted, comparison.norm_next_radiation))
end


"""
    plot_loss_curves(loss_train::Vector, loss_valid::Vector, model = nothing)

Plot the learning curves of some `model` using the per-epoch values of the `loss_train` and the
`valid_train` in a line-chart.
"""
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

    @info "Loading scaler objects for output data"
    target_transformer = FileIO.load(joinpath(ARTIFACT_DIR, "artifacts.jld2"), "output_scaler") 

    #
    # Training the model
    #

    @info "Training with $(size(xtrain, 1)) samples"
    epochs = 10

    @time model, loss_train, loss_valid = train(
        xtrain = Array(permutedims(xtrain)),
        ytrain = Array(permutedims(ytrain[:, :norm_next_radiation])),
        xvalid = Array(permutedims(xvalid)),
        yvalid = Array(permutedims(yvalid[:, :norm_next_radiation])),
        epochs = epochs,
        use_gpu = true,
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
