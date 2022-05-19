# One Explanation to Rule them All -- Ensemble Consistent Explanations

This repository contains the implementation of the methods proposed in the paper [One Explanation to Rule them All -- Ensemble Consistent Explanations](paper.pdf) by André Artelt, Stelios Vrachimis, Demetrios Eliades, Marios Polycarpou and Barbara Hammer.

The implementation of the experiments as well as the implementation of the proposed *ensemble consistent counterfactual explanations* can be found in the folder [Implementation](Implementation/).

## Abstract

Transparency is a major requirement of modern AI based decision making systems deployed in real world. A popular approach for achieving transparency is by means of explanations.
A wide variety of different explanations have been proposed for single decision making systems. 
In practice it is often the case to have a set (i.e. ensemble) of decisions that are used instead of a single decision only, in particular in complex systems. Unfortunately, explanation methods for single decision making systems are not easily applicable to ensembles -- i.e. they would yield an ensemble of individual explanations which are not necessarily consistent, hence less useful and more difficult to understand than a single consistent explanation of all observed phenomena.
We propose a novel concept for consistently explaining an ensemble of decisions locally with a single explanation -- we introduce a formal concept, as well as a specific implementation using counterfactual explanations.

## Details

### Ensemble consistent counterfactual explanations

See `ensemble_consistent_counterfactuals.py`.

### Fault detection

See `FaultDetection.py` and `model.py`.

### Experiments

See `faulty_sensor_identifier.py`.

### Data loader and generator

Loader for simulated data: `FaultySensorData.py` and `LeakySensorData.py`.

Generating the explanations (input to the fault localization method): `generate_l-town-cfsignature_dataset_baseline.py` and `generate_l-town-cfsignature_dataset.py`.

## Data

Datasets (sensor fault configurations, .inp files, etc.) will be published later.

## Requirements

- Python3.6
- Packages as listed in `Implementation/REQUIREMENTS.txt`

## License

MIT license - See [LICENSE](LICENSE)

## How to cite

You can cite the version on [arXiv](https://arxiv.org/abs/2205.08974)
