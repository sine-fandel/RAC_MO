import pickle

import numpy as np

import statistics

from deap import base, gp

from env.simulator.code.simulator.io.loading import load_init_env_data, load_container_data, load_os_data,\
                                                    load_test_container_data, load_test_os_data

from env.simulator.code.simulator import SimulatorState, Simulator

from env.simulator.code.simulator.metrics.pm import FirstFitPMAllocation

import json

def protectedDiv(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            # x = x.astype(np.float64)
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x

class TestGPHH():
    def __init__(self, result_energy, result_communication_path: str, test_path: str, runs: int, algorithms: int, run=0, best_ind_list="", os_list="") -> None:
        """
        """
        self.algorithms = algorithms
        self.result_energy = result_energy
        self.result_communication_path = result_communication_path
        self.test_path = test_path
        self.runs = runs

        # this line does not make any contribution to the simulation, just generate placeholder for the simulation
        init_containers, init_os, init_pms, init_pm_types, init_vms, init_vm_types = load_init_env_data(0).values()
        # Warm up the simulation
        self.sim_state = SimulatorState(init_pms, init_vms, init_containers, init_os, init_pm_types, init_vm_types)

        self.run = run
        self.best_ind_list = best_ind_list
        self.os_list = os_list

    def test_training_dual(self, task=0):
        toolbox = base.Toolbox()
        
        # dual tree gp
        pset = {"vm": None, "pm": None}
        # # terminal nodes list
        TERMINAL_NODES = {"vm": {"ARG0": "container_cpu", "ARG1": "container_memories", "ARG2": "remaining_cpu_capacity", "ARG3": "remaining_memory_capacity", "ARG4": "vm_cpu_overhead", "ARG5": "vm_memory_overhead", "ARG6": "vm_pm_innerc", "ARG7": "vm_pm_outerc"},
                         "pm": {"ARG0": "vm_cpu_capacity", "ARG1": "vm_memory_capacity", "ARG2": "remaining_cpu_capacity", "ARG3": "remaining_memory_capacity", "ARG4": "pm_cpu_capacity", "ARG5": "pm_memory_capacity", "ARG6": "pm_core", "ARG7": "pm_innerc", "ARG8": "pm_outerc"}}

        for type, item in pset.items():
            if type == "vm":
                pset[type] = gp.PrimitiveSet(type, 8)
                pset[type].renameArguments(**TERMINAL_NODES["vm"])
            else:
                pset[type] = gp.PrimitiveSet(type, 9)
                pset[type].renameArguments(**TERMINAL_NODES["pm"])
            pset[type].addPrimitive(np.add, 2)
            pset[type].addPrimitive(np.subtract, 2)
            pset[type].addPrimitive(np.multiply, 2)
            pset[type].addPrimitive(protectedDiv, 2)

        toolbox.register("compile", gp.compile, pset=pset)

        if len(self.best_ind_list) == 1:
            best_individual = self.best_ind_list[0].replace("'", "\"")
        else:
            best_individual = self.best_ind_list[task].replace("'", "\"")
        best_individual = json.loads(best_individual)
        best_sizes = [24000, 20000]
        
        sim = Simulator(self.sim_state, test=True)
        func0 = toolbox.compile(expr=best_individual["vm"], pset=pset["vm"])
        func1 = toolbox.compile(expr=best_individual["pm"], pset=pset["pm"])

        current_timestamp = sim.current_timestamp
        # load test data
        if len(self.os_list) == 1:
            os = self.os_list[0]
        else:
            os = self.os_list[task]
        input_containers, _, application_reverse = load_test_container_data(run=self.run, os=os)
        input_os = load_test_os_data(0, os=os)

        timestamp_list = []
        energy_list = []
        communication_list = []
        if self.algorithms == 1 or self.algorithms == 2 \
            or self.algorithms == 4:
            # min-cut
            # min-cut + resource
            # min-cut + without
            for app, clist in application_reverse.items():
                timestamp_list.append(current_timestamp)
                energy_list.append(sim.running_energy_consumption)
                communication_list.append(sim.running_communication_overhead)
                container_clusters, P, _ = app.min_cut(input_containers, input_os, best_sizes[0], best_sizes[1])
                num = 0
                for row in container_clusters.iterrows():
                    # print(row)
                    vm_selection = sim.vm_selection(func0, app, P[num], row[1][0 : 3], row[1][3], 2)
                    cid_list = []
                    for p in range(len(P[num])):
                        # print(app.vector_id_list[P[num][p]])
                        cid_list.append(app.vector_id_list[P[num][p]]) 
    
                    sim.step_first_layer(vm_selection, cid_list, row[1][0 : 3], row[1][3])
                    if sim.to_allocate_vm_data != None:
                        pm_selection = sim.pm_selection(func1, app, P[num], 2)
                        sim.step_second_layer(pm_selection, cid_list, row[1][0 : 3], row[1][3])
                    num += 1

                # update the communication overhead information
                sim.update_current_communication(app)
                if sim.current_timestamp != current_timestamp:
                    previous_timestamp = current_timestamp
                    current_timestamp = sim.current_timestamp
                    sim.update_total_communication(previous_timestamp, current_timestamp)

            pm_actual_usage = np.array(sim.state.pm_actual_usage)
        else:
            # without
            # communication
            for app, clist in application_reverse.items():
                timestamp_list.append(current_timestamp)
                energy_list.append(sim.running_energy_consumption)
                communication_list.append(sim.running_communication_overhead)
                for i in clist:
                    vm_selection = sim.vm_selection(func0, app, [], input_containers.iloc[i], self.input_os.iloc[i]["os-id"], 2)
                    sim.step_first_layer(vm_selection, [i], input_containers.iloc[i].to_list(), self.input_os.iloc[i]["os-id"])

                    if sim.to_allocate_vm_data != None:
                        pm_selection = sim.pm_selection(func1, app, [], 2)
                        sim.step_second_layer(pm_selection, [i], input_containers.iloc[i].to_list(), self.input_os.iloc[i]["os-id"])

                # update the communication overhead information
                sim.update_current_communication(app)
                if sim.current_timestamp != current_timestamp:
                    previous_timestamp = current_timestamp
                    current_timestamp = sim.current_timestamp
                    sim.update_total_communication(previous_timestamp, current_timestamp)
                    
            pm_actual_usage = np.array(sim.state.pm_actual_usage)

        pm_cpu_utilization = pm_actual_usage[:, 0] / pm_actual_usage[:, -2]
        mean_pm_cpu_utilization = np.mean(pm_cpu_utilization)
        load_balancing = np.sum(np.abs(pm_cpu_utilization - mean_pm_cpu_utilization)) / pm_cpu_utilization.shape[0]
        print(f"{task} energy = {sim.running_energy_consumption}")
        print(f"{task} communication = {sim.running_communication_overhead}")
        print(f"{task} load balancing = {load_balancing}")
    
        return sim.running_energy_consumption, sim.running_communication_overhead, \
                load_balancing


    # def test_training_ob_dual(self, task=0):
    #     toolbox = base.Toolbox()
        
    #     # dual tree gp
    #     pset = {"vm": None, "pm": None}
    #     # # terminal nodes list
    #     TERMINAL_NODES = {"vm": {"ARG0": "container_cpu", "ARG1": "container_memories", "ARG2": "remaining_cpu_capacity", "ARG3": "remaining_memory_capacity", "ARG4": "vm_cpu_overhead", "ARG5": "vm_memory_overhead", "ARG6": "vm_pm_innerc", "ARG7": "vm_pm_outerc"},
    #                      "pm": {"ARG0": "vm_cpu_capacity", "ARG1": "vm_memory_capacity", "ARG2": "remaining_cpu_capacity", "ARG3": "remaining_memory_capacity", "ARG4": "pm_cpu_capacity", "ARG5": "pm_memory_capacity", "ARG6": "pm_core", "ARG7": "pm_innerc", "ARG8": "pm_outerc"}}

    #     for type, item in pset.items():
    #         if type == "vm":
    #             pset[type] = gp.PrimitiveSet(type, 8)
    #             pset[type].renameArguments(**TERMINAL_NODES["vm"])
    #         else:
    #             pset[type] = gp.PrimitiveSet(type, 9)
    #             pset[type].renameArguments(**TERMINAL_NODES["pm"])
    #         pset[type].addPrimitive(np.add, 2)
    #         pset[type].addPrimitive(np.subtract, 2)
    #         pset[type].addPrimitive(np.multiply, 2)
    #         pset[type].addPrimitive(protectedDiv, 2)

    #     toolbox.register("compile", gp.compile, pset=pset)

    #     if len(self.best_ind_list) == 1:
    #         best_individual = self.best_ind_list[0].replace("'", "\"")
    #     else:
    #         best_individual = self.best_ind_list[task].replace("'", "\"")
    #     best_individual = json.loads(best_individual)
    #     best_sizes = [24000, 20000]
        
    #     sim = Simulator(self.sim_state, test=True)
    #     func0 = toolbox.compile(expr=best_individual["vm"], pset=pset["vm"])
    #     func1 = toolbox.compile(expr=best_individual["pm"], pset=pset["pm"])

    #     current_timestamp = sim.current_timestamp
    #     # load test data
    #     input_containers, _, application_reverse = load_test_container_data(run=self.run, os=self.os_list[task])
    #     input_os = load_test_os_data(0, os=self.os_list[task])
    #     # print(input_containers)
    #     if self.algorithms == 0 or self.algorithms == 1 \
    #         or self.algorithms == 3 or self.algorithms == 4:
    #         sel_type = 0
    #     else:
    #         sel_type = 1

    #     timestamp_list = []
    #     energy_list = []
    #     communication_list = []
    #     if self.algorithms == 1 or self.algorithms == 2 \
    #         or self.algorithms == 4:
    #         # min-cut
    #         # min-cut + resource
    #         # min-cut + without
    #         for app, clist in application_reverse.items():
    #             timestamp_list.append(current_timestamp)
    #             energy_list.append(sim.running_energy_consumption)
    #             communication_list.append(sim.running_communication_overhead)
    #             container_clusters, P, _ = app.min_cut(input_containers, input_os, best_sizes[0], best_sizes[1])
    #             num = 0
    #             for row in container_clusters.iterrows():
    #                 # print(row)
    #                 vm_selection = sim.vm_selection(func0, app, P[num], row[1][0 : 3], row[1][3], 2)
    #                 cid_list = []
    #                 for p in range(len(P[num])):
    #                     # print(app.vector_id_list[P[num][p]])
    #                     cid_list.append(app.vector_id_list[P[num][p]]) 
    
    #                 sim.step_first_layer(vm_selection, cid_list, row[1][0 : 3], row[1][3])
    #                 if sim.to_allocate_vm_data != None:
    #                     pm_selection = sim.pm_selection(func1, app, P[num], 2)
    #                     sim.step_second_layer(pm_selection, cid_list, row[1][0 : 3], row[1][3])
    #                 num += 1

    #             # update the communication overhead information
    #             sim.update_current_communication(app)
    #             if sim.current_timestamp != current_timestamp:
    #                 previous_timestamp = current_timestamp
    #                 current_timestamp = sim.current_timestamp
    #                 sim.update_total_communication(previous_timestamp, current_timestamp)

    #         pm_actual_usage = np.array(sim.state.pm_actual_usage)
    #     else:
    #         # without
    #         # communication
    #         for app, clist in application_reverse.items():
    #             timestamp_list.append(current_timestamp)
    #             energy_list.append(sim.running_energy_consumption)
    #             communication_list.append(sim.running_communication_overhead)
    #             for i in clist:
    #                 vm_selection = sim.vm_selection(func0, app, [], input_containers.iloc[i], self.input_os.iloc[i]["os-id"], 2)
    #                 sim.step_first_layer(vm_selection, [i], input_containers.iloc[i].to_list(), self.input_os.iloc[i]["os-id"])

    #                 if sim.to_allocate_vm_data != None:
    #                     pm_selection = sim.pm_selection(func1, app, [], 2)
    #                     sim.step_second_layer(pm_selection, [i], input_containers.iloc[i].to_list(), self.input_os.iloc[i]["os-id"])

    #             # update the communication overhead information
    #             sim.update_current_communication(app)
    #             if sim.current_timestamp != current_timestamp:
    #                 previous_timestamp = current_timestamp
    #                 current_timestamp = sim.current_timestamp
    #                 sim.update_total_communication(previous_timestamp, current_timestamp)
                    
    #         pm_actual_usage = np.array(sim.state.pm_actual_usage)

    #     # print(f"Mostly used VM is {max_vm}, which used {max_count} times")
    #     print(f"{task} energy = {sim.running_energy_consumption}")
    #     print(f"{task} communication = {sim.running_communication_overhead}")
    
    #     return sim.running_energy_consumption, sim.running_communication_overhead, \
    #             np.mean(1 - pm_actual_usage[:, 0] / pm_actual_usage[:, -2])
