import os
import sys
import numpy as np
import pandas as pd


def eval_anomaly_detection(suspicious_time_points, faults_time, fault_labels_all_test):
    test_times = range(len(fault_labels_all_test))
    test_minus_pred = list(set(test_times) - set(suspicious_time_points))

    # Compute detection delay
    detection_delay = list(faults_time).index(suspicious_time_points[0])

    # Compute TPs, FPs, etc. for every point in time when a fault is present
    TP = np.sum([t in faults_time for t in suspicious_time_points]) / len(suspicious_time_points)
    FP = np.sum([t not in faults_time for t in suspicious_time_points]) / len(suspicious_time_points)
    FN = np.sum([t in faults_time for t in test_minus_pred]) / len(test_minus_pred)
    TN = np.sum([t not in faults_time for t in test_minus_pred]) / len(test_minus_pred)

    # Export results
    return {"detection_delay": detection_delay, "tp": TP, "fp": FP, "fn": FN, "tn": TN}


if __name__ == "__main__":
    # Parse arguments
    if len(sys.argv) != 4:
        print("Usage: <data_path_in> <ground_truth.xlsx> <eval_anomaly_detection_out.csv>")
        os._exit(1)
    data_path_in = sys.argv[1]
    ground_truth_in = sys.argv[2]
    eval_anomaly_detection_out = sys.argv[3]

    # Load ground truth
    df_sensorfaults = pd.read_excel(ground_truth_in, sheet_name="Sheet1", engine='openpyxl')
    df_sensorfaults = df_sensorfaults[["scenario", "nodeid_linkid"]]
    scenarios_to_sensor = {}
    for _, row in df_sensorfaults.iterrows():
        scenarios_to_sensor[int(row["scenario"])] = row["nodeid_linkid"].strip()

    # Load counterfactuals
    X = []
    X_orig = []
    y = []

    anomaly_detection_evaluations = []

    files_in = list(filter(lambda z: z.endswith("_sensor_fault.npz") and z.startswith("cfsignature"), os.listdir(data_path_in)))
    for file_in in files_in:
        scenario_id = int(file_in.replace("cfsignature_", "").replace("_sensor_fault.npz", ""))
        data = np.load(os.path.join(data_path_in, file_in), allow_pickle=True)
        X_cf_, X_orig_ = data["cfsignature"], data["faultyinput"]
        
        data_ex = np.load(os.path.join(data_path_in, "1_sensor_fault.npz"))
        sensors = list(data_ex["flow_nodes"]) + list(data_ex["pressure_nodes"])
        #print(sensors)
        suspicious_time_points, faults_time, fault_labels_all_test = data_ex["suspicious_time_points"], data_ex["faults_time"], data_ex["fault_labels_all_test"]

        anomaly_detection_eval = eval_anomaly_detection(suspicious_time_points, faults_time, fault_labels_all_test)
        anomaly_detection_eval["scenario"] = scenario_id

        anomaly_detection_evaluations.append(anomaly_detection_eval)

        X.append(X_cf_)
        X_orig.append(X_orig_)
        y.append(sensors.index(scenarios_to_sensor[scenario_id]))


    # Post process data -> aggregate explanation in a normalized histogram
    X_final, y_final = [], []
    for i in range(len(X)):
		# Treat each alarm separately
		for x_sample, x_orig in zip(X[i], X_orig[i]):
			deltas = []
			for x_ in x_sample:
				d = np.abs(x_orig - x_)
				for k in range(x_.shape[0]):
				if x_[k] == 0.0:    # Target index
					d[k] = 0.
				deltas.append(d)
			delta = np.sum(deltas, axis=0)

			fingerprint = delta
			fingerprint = fingerprint / np.sum(fingerprint)

			X_final.append(fingerprint)
			y_final.append(y[i])

    # Try to identify faulty sensors
    y_pred = []
    for i in range(len(X_final)):
        deltacf = X_final[i]
        y_pred_idx = np.argmax(np.abs(deltacf))

        y_pred.append(y_pred_idx)
    
    # Evaluation of fault localization
    print(y_pred, y_final)
    r = [y_pred[i] == y_final[i] for i in range(len(y_pred))]
    print(f"Faulty sensor identification/localization -- Accuracy: ${np.round(np.mean(r), 2)} \pm {np.round(np.var(r), 2)}$")

    # Evaluation of anomaly detection
    df_anomaly_detection = pd.DataFrame(anomaly_detection_evaluations)
    df_anomaly_detection.to_csv(eval_anomaly_detection_out, index=False)

    print(f"True Positives & $ {df_anomaly_detection['tp'].mean()} \pm {df_anomaly_detection['tp'].var()}$")
    print(f"True Negatives & $ {df_anomaly_detection['tn'].mean()} \pm {df_anomaly_detection['tn'].var()}$")
    print(f"False Positives & $ {df_anomaly_detection['fp'].mean()} \pm {df_anomaly_detection['fp'].var()}$")
    print(f"False Negatives & $ {df_anomaly_detection['fn'].mean()} \pm {df_anomaly_detection['fn'].var()}$")
    print(f"Detection delay & $ {df_anomaly_detection['detection_delay'].mean()} \pm {df_anomaly_detection['detection_delay'].var()}$")
