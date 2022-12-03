using MLJ
using Flux
using BSON
using Dates
using FileIO
using Random
using DataFrames


const MODEL_DIR = joinpath(pwd(), "..", "models")
const ARTIFACT_DIR = joinpath(pwd(), "..", "artifacts")


function fetch_most_recent_model(; subject_to::DateTime)
    all_models = readdir(joinpath(MODEL_DIR))

    # calculate the "distence" between the model timestamps and the "subject_to" param
    model_datetimes = map(m -> DateTime(split(m, r"_|\.")[2], "Y-m-d-H-M"), all_models)
    differences = subject_to .- model_datetimes

    BSON.@load joinpath(MODEL_DIR, all_models[argmin(differences)]) model
    return model
end


function build_comparison(model, x, y; output_scaler) :: DataFrame
    comparison = copy(y)

    comparison.norm_predicted = model(Array(x)')[1, :]
    comparison.predicted = inverse_transform(output_scaler, comparison.norm_predicted)

    return comparison
end


function general_evaluation(data::DataFrame) :: Nothing
    @info "MAE: " * string(Flux.Losses.mae(data.next_radiation, data.predicted))
    @info "RMSE: " * string(sqrt(Flux.Losses.mse(data.next_radiation, data.predicted)))
end


function site_specific_evaluation(data::DataFrame) :: Nothing
    mae = combine(groupby(data, :id), [:next_radiation, :predicted] => Flux.Losses.mae => :mae)
    rmse = combine(groupby(data, :id), [:next_radiation, :predicted] => ((a, b) -> sqrt(Flux.Losses.mse(a, b))) => :rmse)
    nsamples = combine(groupby(data, :id), nrow => :nsamples)

    comparison = innerjoin(mae, rmse, nsamples; on = :id)

    @info comparison
    @info "Standard Deviation (RMSE) " * string(std(comparison[:, :mae]))
    @info "Standard Deviation (MAE) " * string(std(comparison[:, :rmse])) 
end


function main() :: Nothing

    #
    # Loading the most recent model with respecto to a date
    #

    use_latest_model = true
    datetime = Dates.now()

    @info "Loading the latest model with respect to $(Dates.format(datetime, "Y-m-d H:M"))"
    if use_latest_model
        model = fetch_most_recent_model(subject_to = datetime)
    end

    #
    # Loading the testing artifacts
    #

    @info "Loading the testing datasets"
    xtest, ytest = FileIO.load(joinpath(ARTIFACT_DIR, "artifacts.jld2"), "xtest", "ytest")


    @info "Loading the output scaler"
    output_scaler = FileIO.load(joinpath(ARTIFACT_DIR, "artifacts.jld2"), "output_scaler")

    #
    # Commencing evaluation
    #

    comparison_dataset = build_comparison(model, xtest, ytest; output_scaler = output_scaler) 

    general_evaluation(comparison_dataset)
    return site_specific_evaluation(comparison_dataset)

    return nothing
end


main()
