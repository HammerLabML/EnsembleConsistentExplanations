import os
import sys
import random
import numpy as np
import pandas as pd
import pickle
from joblib import Parallel, delayed

from FaultySensorData import FaultySensorScenario, enumerate_all_faulty_scenarios
from LeakySensorData import LeakyScenario, enumerate_all_leaky_scenarios
from FaultDetection import EnsembleSystem
from ensemble_consistent_counterfactuals import EnsembleConsistentCounterfactuals
from utils import evaluate_fault_detection
from model import MyModel



def compute_priors(sensor_errors):      # Put a prior on sensors with a large deviation from their expected value
    c = np.sum(np.exp(sensor_errors))
    return np.array([(1. / (np.exp(s) / c)) for s in sensor_errors])


def process_scenario(scenario_id, fault_type, path_to_leakage_data, path_to_sensor_fault_data, exp_results_out_path, model_in_path=None):
    # Load data
    if fault_type == "leakage":
        scenario = LeakyScenario(path_to_leakage_data, scenario_id)
    elif fault_type == "sensor_fault":
        scenario = FaultySensorScenario(path_to_sensor_fault_data, scenario_id)
    X_all, Y_all, fault_labels_all, pressure_nodes, flow_nodes = scenario.X, scenario.Y, scenario.fault_labels, scenario.pressure_nodes, scenario.flow_nodes

    t_train_split = 5000    # Use the first 5000 sampels for training (we know that there is no anomaly present in this time period)    

    X_all_train, X_all_test = X_all[:t_train_split,:], X_all[t_train_split:,:]
    Y_all_train, Y_all_test = Y_all[:t_train_split,:], Y_all[t_train_split:,:]
    fault_labels_all_train, fault_labels_all_test = fault_labels_all[:t_train_split], fault_labels_all[t_train_split:]

    # Fit model
    if model_in_path is not None:
        with open(model_in_path, "rb") as f_in:
            ensemble_system = pickle.load(f_in)
    else:
        ensemble_system = EnsembleSystem(MyModel, flow_nodes, pressure_nodes)
        ensemble_system.fit(X_all_train, Y_all_train)


    # Anomaly detection
    suspicious_time_points, sensor_errors = ensemble_system.apply_detector(X_all_test, Y_all_test)

    # Evaluation of anomaly detection
    faults_time = np.where(fault_labels_all_test == 1)[0]
    fault_detection_results = evaluate_fault_detection(suspicious_time_points, faults_time)

    np.savez(os.path.join(exp_results_out_path, f"{scenario_id}_{fault_type}"), faults_time=faults_time, suspicious_time_points=suspicious_time_points, fault_labels_all_test=fault_labels_all_test, X_all_train=X_all_train, Y_all_train=Y_all_train, flow_nodes=flow_nodes, pressure_nodes=pressure_nodes)

    # Remove all false alarms
    suspicious_time_points = list(filter(lambda t: t in faults_time, suspicious_time_points))

    # Compute ensemble consistent explanations to obtain counterfactual signatures of the anomaly
    models_wrapper = []
    for m in ensemble_system.models:
        threshold = m["fault_detector"].threshold
        #threshold = 0
        models_wrapper.append({"model": m["model"].wrap_model(), "feature_id_dropped": m["target_idx"], "threshold": threshold})

    counterfactuals = []
    original_samples = []

    suspicious_time_points_ = suspicious_time_points

    explainer = EnsembleConsistentCounterfactuals()
    explainer.solver_verbosity = False

    for t in suspicious_time_points_:
        mad = None
        try:
            xcf, _ = explainer.build_solve_opt(X_all_test[t,:], Y_all_test[t], models_wrapper, soft_constraints=False)
            counterfactuals.append(xcf)
            original_samples.append(X_all_test[t,:])
        except Exception as ex:
            print(ex)

    # Export data
    np.savez(os.path.join(exp_results_out_path, f"cfsignature_{scenario_id}_{fault_type}"), label=0 if fault_type == "leakage" else 1, cfsignature=counterfactuals, faultyinput=original_samples)


if __name__ == "__main__":
    # Parse args
    if len(sys.argv) < 4:
        print("Usage: <path_to_leakage_data> <path_to_sensor_fault_data> <exp_results_out_path> '<path_to_cached_detector>'")
        os._exit(1)
    path_to_leakage_data = sys.argv[1]
    path_to_sensor_fault_data = sys.argv[2]
    exp_results_out_path = sys.argv[3]
    path_to_cached_detector = None
    if len(sys.argv) == 5:
        path_to_cached_detector = sys.argv[4]

    # Enumerate all scenarios
    leaky_scenarios = enumerate_all_leaky_scenarios(path_to_leakage_data)
    faultysensor_scenarios = enumerate_all_faulty_scenarios(path_to_sensor_fault_data)
    
    # Process all scenarios
    results = Parallel(n_jobs=-2)(delayed(process_scenario)(s_id, "sensor_fault", path_to_leakage_data, path_to_sensor_fault_data, exp_results_out_path, model_in_path=path_to_cached_detector) for s_id in faultysensor_scenarios)

    np.savez(os.path.join(exp_results_out_path, "training_dataset.npz"), X=X_all_train, y=Y_all_train, flow_nodes=flow_nodes, pressure_nodes=pressure_nodes)

