# SENS and Axivity activity-recognition analysis

## Scope
This repository contains the analysis code used to compare within-device and
cross-device activity recognition for SENS and Axivity accelerometers. Three
neural-network architectures were evaluated:

- one-dimensional convolutional neural network (CNN);
- CNN with a bidirectional long short-term memory layer (CNN-BiLSTM);
- CNN-BiLSTM with temporal attention.

Each architecture can use 1-, 3- and 5-s windows, or another configured
length, under mixed or pure conditions. Pure windows contain a single
activity label. Mixed windows are assigned the first value returned by
`pandas.Series.mode()`. A tied mode is therefore retained and resolved by
taking the first modal label, matching the original analysis notebooks. The
six analysis classes are Lie, Sit, Stand, Walk, Stairs and Run.

For current study, we only use 3-s windows and mixed.
## Analysis design
Models were trained separately using Axivity and SENS data. A model trained on
one device was evaluated on held-out data from the same device and on the
paired data from the other device. The four reported directions were therefore
Axivity-to-Axivity, Axivity-to-SENS, SENS-to-SENS and SENS-to-Axivity.

Hyperparameters were selected using development data. Final model selection
used validation macro-F1, and the held-out test data were not used to select
training epochs or hyperparameters. The training data supplied the
channel-specific mean and standard deviation used for z-score normalization.

Class imbalance was addressed by class-weighted focal cross-entropy and
class-dependent training augmentation. Early stopping used validation
macro-F1. There is no fixed random seed in the configuration. A new 32-bit
seed is generated for each `Result` run, printed when the run starts, and
used throughout that run. Trial seeds are recorded in
`hyperparameter_results.csv`.

## Result structure

Independent repetitions are stored under `results/` as `Result`, `Result_1`,
`Result_2`, and subsequent numbered folders. Every repetition contains
`within_device` and `cross_device`. Each model-condition folder contains
exactly:

```text
best_model.pt
channel_occlusion_importance.csv
classification_report.csv
hyperparameter_results.csv
temporal_occlusion_importance.csv
test_predictions.csv
```

The folder names and CSV schemas reproduce those used in the original
analysis notebooks. `best_model.pt` contains the validation-selected PyTorch
state dictionary.

## Repeated-run summary

The training controller and reviewer-output script are independent.
`run_experiments.py` creates only the repeated experiment folders and does not
invoke the reviewer-output script. After all repetitions are complete,
`generate_reviewer_tables.py` can be run separately to read all complete
`Result*` folders. Macro-F1 is reported as the arithmetic mean and sample
standard deviation across independent repetitions. Relative change is
calculated as:

```text
100 x (cross-device mean macro-F1 - within-device mean macro-F1)
      / within-device mean macro-F1
```

For each source device, window length and purity condition, the selected
architecture is the model with relative change closest to zero. The
representative baseline and cross-device confusion matrices use the runs
independently closest to their corresponding condition means. Run identifiers
are not reported in Table S3.

The generated summary files are:

```text
Additional_file_1_cross_device_performance.xlsx
Appendix_Table_A1_window_distribution.xlsx
Appendix_Table_A2_acceleration_variability.xlsx
confusion_matrices/
```

The workbook contains Tables S1-S3 on separate worksheets. Table S3 does not
report the runs selected for the representative matrices. The paired
confusion matrices include every complete run-level within/cross pair and the
representative pairs used in the tables.

`generate_appendix_table_a1.py` calculates activity counts and percentages
for the paired training and test windows. SENS and Axivity counts are
identical because both devices use the same synchronized windows.

`generate_appendix_table_a2.py` calculates the sample standard deviation of
each acceleration axis in every paired window, takes participant-level
medians by activity and device, and averages those medians with equal
participant weight. The workbook includes both Table A2 and the participant
medians used to calculate it.

Both appendix workbooks are generated automatically when
`generate_reviewer_tables.py` runs. Either appendix script can also be run
directly when only that workbook needs to be rebuilt.

## Software

Package versions are pinned in
`requirements.txt`. PyTorch supports CPU execution and CUDA-compatible GPUs.
The source files are:

```text
config.json
experiment_config.py
common.py
model_cnn.py
model_cnn_bilstm.py
model_cnn_bilstm_attention.py
run_experiments.py
generate_reviewer_tables.py
generate_appendix_table_a1.py
generate_appendix_table_a2.py
```

## Configuration

All changeable runtime settings are stored in `config.json`: input and output
paths, file matching, sampling rate, run count, models, window lengths, purity
conditions, training controls, device selection, caching, summary settings
and the hyperparameter search trials. Relative paths are resolved from the
directory containing the configuration file. Random seeds are intentionally
not configured and are generated separately for every `Result` run.
