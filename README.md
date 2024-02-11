## Deep Dense Neural Networks for Solar Radiation Forecasting

This project implements the training and evaluation processes of Multi-Layer Perceptrons to perform short-term solar radiation forecasting from
meteorological data (from the data available, forecasting is performed for values 1h in the future). This implementation also comprehends the data preparation pipeline, from the zipped raw data download from the [National Institute of Meteorology of Brazil](https://portal.inmet.gov.br/dadoshistoricos) to the normalization process, the basic evaluation infrastructure
and a few model baselines to be beaten.

The main objective of this project is to compare the performance of a single neural network fitted in data from geographically diverse data-collection sites against the performance of multi-layer machine learning ensembles fitted with site-specific data (i.e. one fitted model per data-collection site).

The implementation was carried out using Julia v1.8 with the addition of the main packages revolving around the ecosystems of `Flux.jl` and `DataFrames.jl`.

## Baselines

As mentioned, a comparison against a few baselines to predict the solar radiation in the next hour are also carried out, such baselines stablished for the task at hand were:

1. **_Houly Mean_**: the radiation in the next hour t + 1 will be computed as the mean solar radiation value across all training observations grouping by the hour of the day. Example: let the current hour = 10h, the radiation for hour = 11h will be the mean value of all observations which took place at 11h.

1. **_Daily Mean_**: the radiation in the next hour t + 1 will be computed as the mean solar radiation value across all training observations grouping by the day of the year. Example: let the current day of year = 110 (April 20th) and the current hour = 10h, the radiation for hour = 11h  will be the mean value of radiation across all observations in the same day of year (in this case the 110th day).

1. **_Date-Time Mean_**: analog to the *Daily Mean* and *Hourly Mean* baselines, in this baseline the radiation in the next hour t + 1 will be computed as the mean solar radiaiton value across all training observations grouping by the day of the year and the hour of the day.

1. **_Previous Day_**: the radiation in the next hour t + 1 will be computed as the same radiation value at t + 1 in the previous day (when available). Example: let the current day of the year = 110 and the current hour = 10, the solar radiation value for hour = 11h will be the same value for the solar radiation in day of year = 109 at hour = 11.

## Installation and Usage

All the project dependencies are specified in the `Project.toml` file, in order to install them and activate the project use:

```julia
(@1.8) pkg> activate
(dl-solar-sao-paulo) pkg> instantiate .
(dl-solar-sao-paulo) pkg> # project is ready to be executed, just exit the package manager mode
```

The scripts contained under `src/` should be executed in the following order:
1. `data_pipeline.jl`
1. `model_training.jl`
1. `model_evaluation.jl`
1. `model_baselines.jl`

## Project Organization

The contents, scripts and artifacts of the project are organized as follows:

| Directory/File | Contents |
|----------------|----------|
| `artifacts/`[^note] | Holds the `artifacts.jld2` file with the scaled data for training validation and testing as well as other artifacts such as performance plots. |
| `data/`[^note] | Contains both raw and processed data. |
| `models/`[^note] | Contains the models fitted by `src/model_training.jl`. |
| `src/` | Contains the julia source code. |
| `src/data_pipeline.jl` | Implementation for the data download, preprocessing and separation. The final contents are stored in `artifacts/` and can be used directly in the training routines.|
| `src/model_training.jl` | Creates and trains the neural network using the preprocessed data. After training the models are stored in the `models/` directory.|
| `src/model_evaluation.jl` | Loads the specified model from `models/` for evaluation. |

[^note]: Created after the first execution of the scripts.
