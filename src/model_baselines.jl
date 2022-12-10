using CSV
using Flux
using Dates
using FileIO
using DataFrames
using Statistics
using ShiftedArrays
using BenchmarkTools


const ARTIFACT_DIR = joinpath(pwd(), "..", "artifacts")


abstract type Baseline end


"""
    HourlyMean(; site_specific::Bool)

The prediction for the target variable for t+1 will be the mean value for the target
in the hours in provided data.
"""
mutable struct HourlyMean <: Baseline
    lookup_table::Union{AbstractDataFrame, Nothing}
    site_specific::Bool
end
HourlyMean(; site_specific = false) = HourlyMean(nothing, site_specific)

function fit!(baseline::HourlyMean, X::DataFrame) :: Nothing
    agg_cols = baseline.site_specific ? [:id, :hour] : [:hour]
    baseline.lookup_table = combine(groupby(X, agg_cols), :next_radiation => mean => :baseline)
    return nothing
end

function predict(baseline::HourlyMean, X::DataFrame)
    merge_cols = baseline.site_specific ? [:id, :hour] : [:hour]
    return leftjoin(X, baseline.lookup_table; on = merge_cols)
end


"""
    DailyMean(; site_specific::Bool)

The prediction for the target variable for t+1 will be the mean value for the target
in the days of the year in `X`.
"""
mutable struct DailyMean <: Baseline
    lookup_table::Union{AbstractDataFrame, Nothing}
    site_specific::Bool
end
DailyMean(; site_specific = false) = DailyMean(nothing, site_specific)

function fit!(baseline::DailyMean, X::DataFrame) :: Nothing
    agg_cols = baseline.site_specific ? [:id, :doy] : [:doy]
    baseline.lookup_table = combine(groupby(X, agg_cols), :next_radiation => mean => :baseline)
    return nothing
end

function predict(baseline::DailyMean, X::DataFrame)
    merge_cols = baseline.site_specific ? [:id, :doy] : [:doy]
    return leftjoin(X, baseline.lookup_table; on = merge_cols)
end


"""
    DateTimeMean(; site_specific::Bool)

The prediction for the target variable for t+1 will be the mean value for the target
in the days of the year in `X`.
"""
mutable struct DateTimeMean <: Baseline
    lookup_table::Union{AbstractDataFrame, Nothing}
    site_specific::Bool
end
DateTimeMean(; site_specific = false) = DateTimeMean(nothing, site_specific)

function fit!(baseline::DateTimeMean, X::DataFrame) :: Nothing
    agg_cols = baseline.site_specific ? [:id, :doy, :hour] : [:doy, :hour]
    baseline.lookup_table = combine(groupby(X, agg_cols), :next_radiation => mean => :baseline)
    return nothing
end

function predict(baseline::DateTimeMean, X::DataFrame)
    merge_cols = baseline.site_specific ? [:id, :doy, :hour] : [:doy, :hour]
    return leftjoin(X, baseline.lookup_table; on = merge_cols)
end


"""
    PreviousDay()

The value for the target in t+1 will be the same value for the target in the same hour
of the previous day. This baseline is inherently site-specific.
"""
mutable struct PreviousDay <: Baseline
    lookup_table::Union{AbstractDataFrame, Nothing}
end
PreviousDay() = PreviousDay(nothing)

function fit!(baseline::PreviousDay, X::DataFrame) :: Nothing

    # similar dataframe, but copying no rows from it
    baseline.lookup_table = similar(X, 0)

    # base dataframe for the 24h window alignment
    start_date = DateTime(minimum(X.year)) + Day(minimum(X.doy) - 1)
    end_date = DateTime(maximum(X.year)) + Day(maximum(X.doy) - 1)
    datearray = start_date:Hour(1):end_date

    fulltime_df = DataFrame(
        hour = hour.(datearray),
        doy = dayofyear.(datearray),
        year = year.(datearray),
    )

    for site in groupby(X, :id)

        filled_df = leftjoin(fulltime_df, site, on = [:hour, :doy, :year])

        # this sort operation is here to make sure that the correct chronological order
        # is kept after the join. The fact that the rows in the resulting dataframe get
        # somehow reordered makes absolutely no sense to me
        sort!(filled_df, [:year, :doy, :hour])

        # now that every day worth of observations has the exact same number of hours
        # we can safely shift the values for the radiation
        # NOTE that here we are shifting 23 hours (not 24) because we heed to the the radiation
        # in the *next hour*
        filled_df[!, :baseline] = ShiftedArrays.lag(filled_df.next_radiation, 23)

        # inserting back the :id column not present in `fulldate_df`
        filled_df[!, :id] .= first(site.id)

        append!(baseline.lookup_table, filled_df, cols = :union)
    end

    dropmissing!(baseline.lookup_table)

    return nothing
end

function predict(baseline::PreviousDay, X::DataFrame)
    return baseline.lookup_table
end


function evaluate_on_general_metrics(data::DataFrame) :: Dict
    df = dropmissing(data)
    return Dict(
        "mae" => Flux.Losses.mae(df.next_radiation, df.baseline),
        "rmse" => sqrt(Flux.Losses.mse(df.next_radiation, df.baseline)),
    )
end


function main()
    ytrain, ytest = FileIO.load(joinpath(ARTIFACT_DIR, "artifacts.jld2"), "ytrain", "ytest")

    baseline_results = DataFrame(
        :baseline => [],
        :mae => [],
        :rmse => [],
        :site_specific => [],
    )

    for baseline in [PreviousDay, HourlyMean, DailyMean, DateTimeMean]
        baseline_models = [baseline()]

        if hasproperty(baseline_models[1], :site_specific)
            push!(baseline_models, baseline(site_specific = true))
        end

        for model in baseline_models
            # ytrain here is just a name convention, the columns that we need to fit the baseline
            # models match the ones in ytrain.
            fit!(model, ytrain)
            baseline_predictions = predict(model, ytest)

            metrics = evaluate_on_general_metrics(baseline_predictions)

            site_specific = hasproperty(model, :site_specific) ? model.site_specific : nothing 
            push!(baseline_results, (typeof(model), metrics["mae"], metrics["rmse"], site_specific))
        end
    end

    sort!(baseline_results, :rmse)
    @info baseline_results
end

@time main()
