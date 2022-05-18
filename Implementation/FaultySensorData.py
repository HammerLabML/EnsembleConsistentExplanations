import os
import numpy as np
import pandas as pd

from ScenarioPreprocessing import ScenarioPreprocessing


class FaultySensorScenario(ScenarioPreprocessing):
    def __init__(self, path_to_data, scenario_id, **kwds):
        super().__init__(**kwds)

        self.path_to_data = path_to_data
        self.scenario_id = scenario_id

        self.faulty_sensor_id = None
        self.pressure_nodes = None
        self.flow_nodes = None
        self.X = None
        self.Y = None
        self.feat_desc = None
        self.fault_labels = None

        self.__load()

    def __load(self):
        # Parse information about the fault
        fault_info_file = list(filter(lambda z: z.endswith(".xlsx"), os.listdir(os.path.join(self.path_to_data, f"scenario{self.scenario_id}", "WithoutSensorFaults"))))[0]
        df_fault_info = pd.read_excel(os.path.join(self.path_to_data, f"scenario{self.scenario_id}", "WithoutSensorFaults", fault_info_file), sheet_name="Info", engine='openpyxl')
        for _, row in df_fault_info.iterrows():
            if row["Description"] == "Fault Start":
                faulty_sensor_start = row["Value"]
            elif row["Description"] == "Fault End":
                faulty_sensor_end = row["Value"]
            elif row["Description"] == "NODE ID":
                self.faulty_sensor_id = row["Value"].replace("pressure", "")

        # Parse pressure measurements
        df_pressures = pd.read_excel(os.path.join(self.path_to_data, f"scenario{self.scenario_id}", "Measurements.xlsx"), sheet_name="Pressures (m)", engine='openpyxl')

        self.pressure_nodes = list(df_pressures.columns);self.pressure_nodes.remove("Timestamp")

        pressures_per_node = {}
        for node_id in self.pressure_nodes:
            pressures_per_node[node_id] = df_pressures[[node_id]].to_numpy().flatten()

        # Parse flow measurements
        df_flows = pd.read_excel(os.path.join(self.path_to_data, f"scenario{self.scenario_id}", "Measurements.xlsx"), sheet_name="Flows (m3_h)", engine='openpyxl')

        self.flow_nodes = list(df_flows.columns);self.flow_nodes.remove("Timestamp")
        flows_per_node = {}
        for node_id in self.flow_nodes:
            flows_per_node[node_id] = df_flows[[node_id]].to_numpy().flatten()

        # Create labels
        df_labels = df_pressures[["Timestamp"]].copy()
        df_labels["label"] = 0

        indices = df_labels[(df_labels["Timestamp"] >= faulty_sensor_start) & (df_labels["Timestamp"] <= faulty_sensor_end)].index
        for idx in indices:
            df_labels["label"].loc[idx] = 1

        labels = df_labels["label"].to_numpy().flatten()

        # Build numpy arrays
        y = labels
        X = np.vstack([pressures_per_node[n] for n in self.pressure_nodes] + [flows_per_node[n] for n in self.flow_nodes]).T
        self.feat_desc = self.pressure_nodes + self.flow_nodes

        # Apply preprocessing
        self.X, self.Y, self.fault_labels = self.global_preprocessing(X, y)


def enumerate_all_faulty_scenarios(path_to_scenarios):
    return list(map(lambda z: int(z.replace("scenario", "")), filter(lambda z: z.startswith("scenario"), os.listdir(path_to_scenarios))))
