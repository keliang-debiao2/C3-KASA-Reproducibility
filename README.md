C3-KASA
Construct-Specific Cross-Modal Ordinal Scoring with Calibrated Uncertainty for L2 Korean Speaking Assessment

This package contains the C3-KASA model implementation, feature-loading utilities, training and prediction scripts, calibration functions, evaluation metrics, statistical routines, configuration files, data schemas, documentation, tests, and reference records.
Package Contents
Path	Contents
src/c3kasa/	Python modules for modeling, data handling, losses, calibration, metrics, statistics, configuration, file I/O, and random seed control.
scripts/	Command-line scripts for data auditing, speaker-based partitioning, training, prediction, calibration, evaluation, configuration checks, reference-table export, and repository validation.
configs/	YAML configurations for the main model, primary data, outer folds, transfer partitions, over-the-air evaluation planning, and the cross-attention control.
schemas/	JSON schemas for response records, feature records, and over-the-air capture manifests.
docs/	Documentation covering data contracts, output conventions, reproduction procedures, and anonymity checks.
tests/	Checks for model outputs, metric behavior, and speaker-disjoint data partitions.
reference_logs/	JSONL reference records covering data auditing, outer folds, calibration, the parameter-matched control, transfer, over-the-air evaluation, and statistics.
artifacts/reference_metrics/	CSV reference records organized by primary folds, baselines, calibration, the parameter-matched control, transfer, over-the-air evaluation, significance, and qualitative cases.
pyproject.toml	Package metadata, dependency declarations, optional dependency groups, and test configuration.
requirements.txt	Core dependency requirements.
Makefile	Convenience commands.
MANIFEST.sha256	File checksum manifest supplied with the archive.
LICENSE	MIT license for the code.


Model Components
src/c3kasa/model.py contains the following components:
- C3KASA: input projections for five feature streams, a shared representation, construct-specific gated routing, ordinal scoring heads, and evidence-strength heads.
- ConstructRouter: learned stream weighting and construct-specific feature aggregation.
- OrderedOrdinalHead: ordered thresholds, cumulative probabilities, and ordinal class probabilities.
- EvidenceHead: evidence strength, Dirichlet parameters, and an uncertainty score.
- ParameterMatchedCrossAttention: a Transformer-based feature-fusion control with residual processing and construct-specific ordinal heads.
The five feature streams are speech_semantic, asr, pronunciation_prosody, morphosyntax, and prompt_content. The three scoring constructs are delivery, language_use, and content.
Core Python Modules
Module	Included functionality
data.py	Manifest auditing, speaker-grouped fold assignment, train/selection/calibration/test partitioning, feature loading from NumPy files, and a PyTorch dataset.
losses.py	Ordinal negative log-likelihood, concordance correlation loss, evidence-related loss, representation orthogonality, and their joint objective.
calibration.py	Temperature fitting for cumulative ordinal probabilities, probability conversion, score rounding, finite-sample quantiles, and speaker-blocked quantiles.
metrics.py	Quadratic weighted kappa, concordance correlation, Pearson correlation, MAE, RMSE, macro-F1, exact agreement, within-one agreement, severe disagreement, NLL, Brier score, and ECE.
statistics.py	Clustered bootstrap differences, paired cluster-level permutation tests, and Holm adjustment.
config.py	YAML configuration loading and stable configuration hashing.
io.py	JSON and JSONL file-writing helpers.
seed.py	Random seed and deterministic execution settings.


Command-Line Scripts
Script	Function
audit_data.py	Inspect a response manifest and write an audit record.
make_splits.py	Generate speaker-grouped outer folds and a partition matrix.
train_outer_fold.py	Train C3-KASA for a selected fold and seed; save training logs, run metadata, and a model checkpoint.
predict_outer_fold.py	Load a checkpoint and export predictions for a selected partition role, with optional calibration.
calibrate_outer_fold.py	Fit calibration parameters from prediction records.
evaluate_outer_fold.py	Calculate metrics from prediction files.
run_parameter_matched_control.py	Write a control-plan record from the cross-attention configuration.
run_transfer.py	Check transfer-partition configuration consistency and export the checked configuration.
run_ota.py	Check over-the-air evaluation configuration consistency and export an evaluation plan.
run_statistics.py	Write a statistical-analysis plan specifying clustering, resampling, and multiple-comparison settings.
build_report_tables.py	Copy the supplied reference-metric CSV files to an output directory.
run_unit_checks.py	Execute the packaged unit-check functions.
validate_repository.py	Check repository text and filenames against configured token rules and parse YAML and JSON files.


Data Interfaces and Schemas
The package uses explicit manifests to connect response metadata and precomputed features to the code.
- Response records: response identifiers, speaker identifiers, prompt identifiers, audio paths, and labels for the three scoring constructs.
- Feature records: response and speaker identifiers, construct labels, and paths to the five feature streams.
- Feature files: .npy or .npz arrays loaded by the dataset utilities.
- Prediction records: response and speaker identifiers, fold and seed identifiers, construct names, labels, class probabilities, expected scores, and uncertainty fields.
- Over-the-air capture records: fields defined by ota_capture_manifest.schema.json.
The archive provides schemas and interface documentation. Audio corpora, precomputed feature datasets, third-party pretrained weights, and trained checkpoints are not bundled.
Reference Files
reference_logs/ and artifacts/reference_metrics/ contain supplied reference records and field structures for inspecting logging and tabular formats. They are separate from outputs generated by new executions.
Documentation and Tests
The docs/ directory contains:
- DATA_CONTRACT.md: manifest fields, feature-file conventions, prediction fields, and data-partition boundaries.
- RESULT_CONTRACT.md: metric aggregation, rounding rules, probability conventions, and reference-record handling.
- REPRODUCTION.md: environment, data preparation, and execution procedures.
- ANONYMITY_CHECKLIST.md: checks for anonymous repository preparation.
The tests/ directory contains test_model.py, test_metrics.py, and test_splits.py.
Dependencies and License
The package metadata specifies Python 3.10 or later and core dependencies on NumPy, pandas, SciPy, scikit-learn, PyYAML, PyTorch, and jsonschema. Optional dependency groups cover development checks and backbone-related libraries.

The code is distributed under the MIT license in LICENSE. External datasets and pretrained models remain subject to their providers' terms.
