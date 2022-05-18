import numpy as np



class ScenarioPreprocessing():
    def __init__(self, time_win_len=4, time_start=100):
        self.time_win_len = time_win_len
        self.time_start = time_start

    def global_preprocessing(self, X, y_labels, sensors_idx=None):
        X_final = []
        Y_final = []
        y_fault = []
        
        if sensors_idx is None:
            sensors_idx = list(range(X.shape[1]))
        
        # Use a sliding time window to construct a labeled data set
        t_index = self.time_start
        time_points = range(len(y_labels))
        i = 0
        while t_index < len(time_points) - self.time_win_len:
            # Grab time window from data stream
            x = X[t_index:t_index+self.time_win_len-1, sensors_idx]

            #######################
            # Feature engineering #
            #######################
            x = np.mean(x,axis=0)  # "Stupid" feature
            X_final.append(x)

            Y_final.append([X[t_index + self.time_win_len-1, n] for n in sensors_idx])

            y_fault.append(y_labels[t_index + self.time_win_len-1])

            t_index += 1  # Note: Overlapping time windows
            i += 1

        X_final = np.array(X_final)
        Y_final = np.array(Y_final)
        y_fault = np.array(y_fault)

        return X_final, Y_final, y_fault
