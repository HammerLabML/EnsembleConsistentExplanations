import numpy as np


def evaluate_fault_detection(faults_time_pred, faults_time_truth, tol=10, false_alarms_tol=2, use_intervals=True):
    false_alarms = 0
    faults_detected = 0
    faults_not_detected = 0

    # Extract intervals in which a fault is present
    intervals = []
    i = 0
    t0 = faults_time_truth[i]
    while i < len(faults_time_truth)-1:
        if faults_time_truth[i + 1] != faults_time_truth[i] + 1:
            intervals.append((t0, faults_time_truth[i]))
            t0 = faults_time_truth[i + 1]
        if not (i + 1 < len(faults_time_truth)-1):
            intervals.append((t0, faults_time_truth[i+1]))

        i += 1

    # Check for false alarms
    for i in range(len(faults_time_pred)):
        t = faults_time_pred[i]
        b = False
        for dt in faults_time_truth:
            if dt - tol <= t and t <= dt + tol:
                b = True
                break
        if b is False:  # False alarm
            if i + false_alarms_tol <= len(faults_time_pred)-1:    # Need a minimum of number of continous alarms for triggering a "real alarm" -- ignore noise!
                if all([t + j == faults_time_pred[i+j] for j in range(false_alarms_tol)]):
                    false_alarms += 1
    
    # Check for detected and undetected faults
    if use_intervals:
        for t0, t1 in intervals:
            b = False
            for t in faults_time_pred:
                if t0 <= t and t <= t1: # TODO: Use tolerance?
                    b = True
                    faults_detected += 1
                    break

            if b is False:
                faults_not_detected += 1
    else:
        for dt in faults_time_truth:
            b = False
            for t in faults_time_pred:
                if dt - tol <= t and t <= dt + tol:
                    b = True
                    faults_detected += 1
                    break
            if b is False:
                faults_not_detected += 1

    return {"false_positives": false_alarms, "true_positives": faults_detected, "false_negatives": faults_not_detected}
