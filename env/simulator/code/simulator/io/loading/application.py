import numpy as np

import env.simulator.code.simulator.config as config 

from typing import Dict, Union, List

from pandas import DataFrame
import pandas as pd
import functools

import copy

import networkx as nx

# DATASET_DIR_LOOKUP = { 'auvergrid' : config.AUVERGRID_DIR,
#                        'bitbrains' : config.BITBRAINS_DIR }

FULL_CONTAINER_COLS = ['cpu', 'memory', 'timestamp']
CONTAINER_COLS = ['cpu', 'memory']
OS_COLS = ['os-id']
PM_TYPE_COLS = ['pm-type-id']
VM_TYPE_COLS = ['vm-type-id']

COLUMN_NAMES_LOOKUP = { 'container-data' : FULL_CONTAINER_COLS,
                        'container' : CONTAINER_COLS,
                        'os' : OS_COLS,
                        'pmType' : PM_TYPE_COLS,
                        'vmType' : VM_TYPE_COLS }

class Application:
    def __init__(self, id_list: list, pattern: int, test_num: int) -> None:
        """ Represent the application as a direct graph
        pattern==0~7: one of the pattern out of 8 application patterns.
        pattern=8: single container
        """
        self.size = len(id_list)           # number of microservices in the application
        self.vector_id_list = np.array(id_list)

        self.test_num = test_num
        self.communication = np.zeros([self.size, self.size], dtype=float)
        self.pattern = pattern

        self.app = nx.DiGraph()

        self.communication_data_volume = [[318.20989166, 445.10047246, 159.09642256,  54.52844991, 74.39986163],
                                        [764.17156665, 92.8829667 , 677.1237676 , 161.71804769, 52.63111215, 68.0477941, 53.82094874, 60.75016662, 61.74171223],
                                        [ 20.54117565, 43.49535301, 160.96986091, 985.17221472, 237.93068686,  97.92186459, 388.68151981],
                                        [146.748153  , 572.9439823 , 354.16353193,  87.32918116, 143.38971983, 243.02994944, 168.91685281, 129.25206447, 290.58602061, 179.93488034,  26.18401661],
                                        [ 725.3920814 , 57.56219731,  115.5015218 , 4136.34083822, 55.57411232,  159.14446437,  283.5878488, 451.3417475, 218.57721579,   59.61189674,  193.88247394,  173.39459053, 589.67150992],
                                        [ 47.42206705, 97.68158879, 104.68103309, 423.5399246 , 180.13457432, 295.74541193,  76.93764343, 293.47236982, 138.02128605, 423.23079475, 148.10475665,  64.35062729, 536.25631279, 151.09888232],
                                        [ 144.2560356, 123.39706504, 310.69017634, 17.79385564, 569.56736434,  519.3216584 ,  116.05284581,  884.02941252, 276.41381169, 1399.8040162 ,  147.11751854, 2371.4532702 , 59.5733879 , 108.87492376, 77.8139596],
                                        [524.10232589,  94.29866323, 121.80943025,  30.23038099, 146.58486912,  40.75775702, 252.63566922, 264.01707983, 147.09779223,  25.78720363, 283.87570654, 526.12373707, 256.97302674,  62.16651809, 132.48150778, 304.29163997, 725.39383566]]

        self.generate_communication()

    def generate_communication(self) -> None:
        """ Generate the communication volumes between microservices
        """
        if self.pattern == 0:
            self.app.add_edge(0, 1, capacity=self.communication_data_volume[0][0])
            self.app.add_edge(1, 2, capacity=self.communication_data_volume[0][1])
            self.app.add_edge(2, 3, capacity=self.communication_data_volume[0][2])
            self.app.add_edge(3, 4, capacity=self.communication_data_volume[0][3])
            self.app.add_edge(4, 5, capacity=self.communication_data_volume[0][4])
            self.communication[0][1] = self.communication_data_volume[0][0]
            self.communication[1][2] = self.communication_data_volume[0][1]
            self.communication[2][3] = self.communication_data_volume[0][2]
            self.communication[3][4] = self.communication_data_volume[0][3]
            self.communication[4][5] = self.communication_data_volume[0][4]
        
        elif self.pattern == 1:
            self.app.add_edge(0, 1, capacity=self.communication_data_volume[1][0])
            self.app.add_edge(1, 2, capacity=self.communication_data_volume[1][1])
            self.app.add_edge(1, 3, capacity=self.communication_data_volume[1][2])
            self.app.add_edge(2, 4, capacity=self.communication_data_volume[1][3])
            self.app.add_edge(3, 5, capacity=self.communication_data_volume[1][4])
            self.app.add_edge(4, 6, capacity=self.communication_data_volume[1][5])
            self.app.add_edge(5, 7, capacity=self.communication_data_volume[1][6])
            self.app.add_edge(6, 8, capacity=self.communication_data_volume[1][7])
            self.app.add_edge(7, 8, capacity=self.communication_data_volume[1][8])
            self.communication[0][1] = self.communication_data_volume[1][0]
            self.communication[1][2] = self.communication_data_volume[1][1]
            self.communication[1][3] = self.communication_data_volume[1][2]
            self.communication[2][4] = self.communication_data_volume[1][3]
            self.communication[3][5] = self.communication_data_volume[1][4]
            self.communication[4][6] = self.communication_data_volume[1][5]
            self.communication[5][7] = self.communication_data_volume[1][6]
            self.communication[6][8] = self.communication_data_volume[1][7]
            self.communication[7][8] = self.communication_data_volume[1][8]
        
        elif self.pattern == 2:
            self.app.add_edge(0, 1, capacity=self.communication_data_volume[2][0])
            self.app.add_edge(1, 2, capacity=self.communication_data_volume[2][1])
            self.app.add_edge(2, 3, capacity=self.communication_data_volume[2][2])
            self.app.add_edge(2, 4, capacity=self.communication_data_volume[2][3])
            self.app.add_edge(3, 5, capacity=self.communication_data_volume[2][4])
            self.app.add_edge(4, 5, capacity=self.communication_data_volume[2][5])
            self.app.add_edge(5, 6, capacity=self.communication_data_volume[2][6])
            self.communication[0][1] = self.communication_data_volume[2][0]
            self.communication[1][2] = self.communication_data_volume[2][1]
            self.communication[2][3] = self.communication_data_volume[2][2]
            self.communication[2][4] = self.communication_data_volume[2][3]
            self.communication[3][5] = self.communication_data_volume[2][4]
            self.communication[4][5] = self.communication_data_volume[2][5]
            self.communication[5][6] = self.communication_data_volume[2][6]
            
        elif self.pattern == 3:
            self.app.add_edge(0, 1, capacity=self.communication_data_volume[3][0])
            self.app.add_edge(1, 2, capacity=self.communication_data_volume[3][1])
            self.app.add_edge(1, 3, capacity=self.communication_data_volume[3][2])
            self.app.add_edge(2, 4, capacity=self.communication_data_volume[3][3])
            self.app.add_edge(2, 5, capacity=self.communication_data_volume[3][4])
            self.app.add_edge(3, 6, capacity=self.communication_data_volume[3][5])
            self.app.add_edge(4, 7, capacity=self.communication_data_volume[3][6])
            self.app.add_edge(5, 7, capacity=self.communication_data_volume[3][7])
            self.app.add_edge(6, 8, capacity=self.communication_data_volume[3][8])
            self.app.add_edge(7, 9, capacity=self.communication_data_volume[3][9])
            self.app.add_edge(8, 9, capacity=self.communication_data_volume[3][10])
            self.communication[0][1] = self.communication_data_volume[3][0]
            self.communication[1][2] = self.communication_data_volume[3][1]
            self.communication[1][3] = self.communication_data_volume[3][2]
            self.communication[2][4] = self.communication_data_volume[3][3]
            self.communication[2][5] = self.communication_data_volume[3][4]
            self.communication[3][6] = self.communication_data_volume[3][5]
            self.communication[4][7] = self.communication_data_volume[3][6]
            self.communication[5][7] = self.communication_data_volume[3][7]
            self.communication[6][8] = self.communication_data_volume[3][8]
            self.communication[7][9] = self.communication_data_volume[3][9]
            self.communication[8][9] = self.communication_data_volume[3][10]

        elif self.pattern == 4:
            self.app.add_edge(0, 1, capacity=self.communication_data_volume[4][0])
            self.app.add_edge(1, 2, capacity=self.communication_data_volume[4][1])
            self.app.add_edge(1, 3, capacity=self.communication_data_volume[4][2])
            self.app.add_edge(2, 3, capacity=self.communication_data_volume[4][3])
            self.app.add_edge(2, 5, capacity=self.communication_data_volume[4][4])
            self.app.add_edge(3, 6, capacity=self.communication_data_volume[4][5])
            self.app.add_edge(3, 7, capacity=self.communication_data_volume[4][6])
            self.app.add_edge(4, 8, capacity=self.communication_data_volume[4][7])
            self.app.add_edge(5, 8, capacity=self.communication_data_volume[4][8])
            self.app.add_edge(6, 9, capacity=self.communication_data_volume[4][9])
            self.app.add_edge(7, 9, capacity=self.communication_data_volume[4][10])
            self.app.add_edge(8, 10, capacity=self.communication_data_volume[4][11])
            self.app.add_edge(9, 10, capacity=self.communication_data_volume[4][12])
            self.communication[0][1] = self.communication_data_volume[4][0]
            self.communication[1][2] = self.communication_data_volume[4][1]
            self.communication[1][3] = self.communication_data_volume[4][2]
            self.communication[2][4] = self.communication_data_volume[4][3]
            self.communication[2][5] = self.communication_data_volume[4][4]
            self.communication[3][6] = self.communication_data_volume[4][5]
            self.communication[3][7] = self.communication_data_volume[4][6]
            self.communication[4][8] = self.communication_data_volume[4][7]
            self.communication[5][8] = self.communication_data_volume[4][8]
            self.communication[6][9] = self.communication_data_volume[4][9]
            self.communication[7][9] = self.communication_data_volume[4][10]
            self.communication[8][10] = self.communication_data_volume[4][11]
            self.communication[9][10] = self.communication_data_volume[4][12]

        elif self.pattern == 5:
            self.app.add_edge(0, 1, capacity=self.communication_data_volume[5][0])
            self.app.add_edge(1, 2, capacity=self.communication_data_volume[5][1])
            self.app.add_edge(1, 3, capacity=self.communication_data_volume[5][2])
            self.app.add_edge(1, 4, capacity=self.communication_data_volume[5][3])
            self.app.add_edge(2, 5, capacity=self.communication_data_volume[5][4])
            self.app.add_edge(3, 6, capacity=self.communication_data_volume[5][5])
            self.app.add_edge(4, 7, capacity=self.communication_data_volume[5][6])
            self.app.add_edge(5, 8, capacity=self.communication_data_volume[5][7])
            self.app.add_edge(5, 8, capacity=self.communication_data_volume[5][8])
            self.app.add_edge(6, 9, capacity=self.communication_data_volume[5][9])
            self.app.add_edge(7, 10, capacity=self.communication_data_volume[5][10])
            self.app.add_edge(8, 11, capacity=self.communication_data_volume[5][11])
            self.app.add_edge(9, 11, capacity=self.communication_data_volume[5][11])
            self.app.add_edge(10, 11, capacity=self.communication_data_volume[5][11])
            self.communication[0][1] = self.communication_data_volume[5][0]
            self.communication[1][2] = self.communication_data_volume[5][1]
            self.communication[1][3] = self.communication_data_volume[5][2]
            self.communication[1][4] = self.communication_data_volume[5][3]
            self.communication[2][5] = self.communication_data_volume[5][4]
            self.communication[3][6] = self.communication_data_volume[5][5]
            self.communication[4][7] = self.communication_data_volume[5][6]
            self.communication[5][8] = self.communication_data_volume[5][7]
            self.communication[5][8] = self.communication_data_volume[5][8]
            self.communication[6][9] = self.communication_data_volume[5][9]
            self.communication[7][10] = self.communication_data_volume[5][10]
            self.communication[8][11] = self.communication_data_volume[5][11]
            self.communication[9][11] = self.communication_data_volume[5][12]
            self.communication[10][11] = self.communication_data_volume[5][13]

        elif self.pattern == 6:
            self.app.add_edge(0, 1, capacity=self.communication_data_volume[6][0])
            self.app.add_edge(1, 2, capacity=self.communication_data_volume[6][1])
            self.app.add_edge(1, 3, capacity=self.communication_data_volume[6][2])
            self.app.add_edge(1, 4, capacity=self.communication_data_volume[6][3])
            self.app.add_edge(2, 5, capacity=self.communication_data_volume[6][4])
            self.app.add_edge(2, 6, capacity=self.communication_data_volume[6][5])
            self.app.add_edge(3, 7, capacity=self.communication_data_volume[6][6])
            self.app.add_edge(4, 8, capacity=self.communication_data_volume[6][7])
            self.app.add_edge(5, 9, capacity=self.communication_data_volume[6][8])
            self.app.add_edge(6, 9, capacity=self.communication_data_volume[6][9])
            self.app.add_edge(7, 10, capacity=self.communication_data_volume[6][10])
            self.app.add_edge(8, 11, capacity=self.communication_data_volume[6][11])
            self.app.add_edge(9, 12, capacity=self.communication_data_volume[6][12])
            self.app.add_edge(10, 12, capacity=self.communication_data_volume[6][13])
            self.app.add_edge(11, 12, capacity=self.communication_data_volume[6][14])
            self.communication[0][1] = self.communication_data_volume[6][0]
            self.communication[1][2] = self.communication_data_volume[6][1]
            self.communication[1][3] = self.communication_data_volume[6][2]
            self.communication[1][4] = self.communication_data_volume[6][3]
            self.communication[2][5] = self.communication_data_volume[6][4]
            self.communication[2][6] = self.communication_data_volume[6][5]
            self.communication[3][7] = self.communication_data_volume[6][6]
            self.communication[4][8] = self.communication_data_volume[6][7]
            self.communication[5][9] = self.communication_data_volume[6][8]
            self.communication[6][9] = self.communication_data_volume[6][9]
            self.communication[7][10] = self.communication_data_volume[6][10]
            self.communication[8][11] = self.communication_data_volume[6][11]
            self.communication[9][12] = self.communication_data_volume[6][12]
            self.communication[10][12] = self.communication_data_volume[6][13]
            self.communication[11][12] = self.communication_data_volume[6][14]

        elif self.pattern == 7:
            self.app.add_edge(0, 1, capacity=self.communication_data_volume[7][0])
            self.app.add_edge(1, 2, capacity=self.communication_data_volume[7][1])
            self.app.add_edge(1, 3, capacity=self.communication_data_volume[7][2])
            self.app.add_edge(1, 4, capacity=self.communication_data_volume[7][3])
            self.app.add_edge(2, 5, capacity=self.communication_data_volume[7][4])
            self.app.add_edge(2, 6, capacity=self.communication_data_volume[7][5])
            self.app.add_edge(3, 7, capacity=self.communication_data_volume[7][6])
            self.app.add_edge(4, 8, capacity=self.communication_data_volume[7][7])
            self.app.add_edge(4, 9, capacity=self.communication_data_volume[7][8])
            self.app.add_edge(5, 10, capacity=self.communication_data_volume[7][9])
            self.app.add_edge(6, 10, capacity=self.communication_data_volume[7][10])
            self.app.add_edge(7, 11, capacity=self.communication_data_volume[7][11])
            self.app.add_edge(8, 12, capacity=self.communication_data_volume[7][12])
            self.app.add_edge(9, 12, capacity=self.communication_data_volume[7][13])
            self.app.add_edge(10, 13, capacity=self.communication_data_volume[7][14])
            self.app.add_edge(11, 13, capacity=self.communication_data_volume[7][15])
            self.app.add_edge(12, 13, capacity=self.communication_data_volume[7][16])
            self.communication[0][1] = self.communication_data_volume[7][0]
            self.communication[1][2] = self.communication_data_volume[7][1]
            self.communication[1][3] = self.communication_data_volume[7][2]
            self.communication[1][4] = self.communication_data_volume[7][3]
            self.communication[2][5] = self.communication_data_volume[7][4]
            self.communication[2][6] = self.communication_data_volume[7][5]
            self.communication[3][7] = self.communication_data_volume[7][6]
            self.communication[4][8] = self.communication_data_volume[7][7]
            self.communication[4][9] = self.communication_data_volume[7][8]
            self.communication[5][10] = self.communication_data_volume[7][9]
            self.communication[6][10] = self.communication_data_volume[7][10]
            self.communication[7][11] = self.communication_data_volume[7][11]
            self.communication[8][12] = self.communication_data_volume[7][12]
            self.communication[9][12] = self.communication_data_volume[7][13]
            self.communication[10][13] = self.communication_data_volume[7][14]
            self.communication[11][13] = self.communication_data_volume[7][15]
            self.communication[12][13] = self.communication_data_volume[7][16]

        else:
            """ Single container
            """
            pass

    def min_cut(self, containers, os, cpu_threhold=24000, memory_threhold=20000):
        P = []
        container_clusters = []
        if len(self.vector_id_list) == 1:
            index = self.vector_id_list[0]
            P.append([0])
            cpu = containers.iloc[index]["cpu"]
            memory = containers.iloc[index]["memory"]
            timestamp = containers.iloc[index]["timestamp"]
            os_id = os.iloc[index]["os-id"]
            container_clusters.append([cpu, memory, timestamp, os_id])

            container_clusters = pd.DataFrame(container_clusters, columns=["cpu", "memory", "timestamp", "os-id"])
            
            return container_clusters, P
        else:
            P = [list(self.app.nodes())]
            i = 0
            while i < len(P):
                cpu = 0
                memory = 0
                for j in P[i]:
                    cpu += float(containers.iloc[self.vector_id_list[j]]["cpu"])
                    memory += float(containers.iloc[self.vector_id_list[j]]["memory"])
                if cpu <= cpu_threhold and memory <= memory_threhold:
                    i += 1
                    continue
                tp = P[i]
                if len(tp) == 1:
                    i += 1
                    continue
                P.remove(P[i])
                start = tp[0]
                end = tp[-1]
                cut_value, partition = nx.minimum_cut(self.app.subgraph(tp), start, end, capacity="capacity")
                p0 = list(self.app.subgraph(partition[0]).nodes())
                p1 = list(self.app.subgraph(partition[1]).nodes())
                P.append(p0)
                P.append(p1)

                i = 0

            # print(P)
            for nodes in P:
                cpu = 0
                memory = 0
                for n in nodes:
                    index = self.vector_id_list[n]
                    cpu += float(containers.iloc[index]["cpu"])
                    memory += float(containers.iloc[index]["memory"])
                    timestamp = float(containers.iloc[index]["timestamp"])
                    os_id = os.iloc[index]["os-id"]
                
                container_clusters.append([cpu, memory, timestamp, os_id])
            
            container_clusters = pd.DataFrame(container_clusters, columns=["cpu", "memory", "timestamp", "os-id"])
            
            P_container_id = []
            for p in P:
                container_id = []
                for index in p:
                    container_id.append(self.vector_id_list[index])
                P_container_id.append(container_id)

            return container_clusters, P, P_container_id

    def loose_min_cut(self) -> None:
        """ cut the application into two partitions
        """
        shape = self.communication.shape[0]
        selected_nodes = np.zeros(shape)
        num_cuts = 0                    # count the number of cuts
        
        super_nodes = []
        communication = copy.deepcopy(self.communication)
        while num_cuts < 4  and len(self.vector_id_list) > 1:
            index = np.unravel_index(communication.argmax(), communication.shape)
            i, j = index[0], index[1]

            con_id_i, con_id_j = self.vector_id_list[i], self.vector_id_list[j]
            
            communication[:, i] += communication[:, j]
            communication[i, i] = 0
            communication[j, j] = 0
            communication[:, j] = 0
            communication[i, :] += communication[j, :]
            communication[i, i] = 0
            communication[j, j] = 0
            communication[j, :] = 0
            flag = 0
            if selected_nodes[i] == 1 and selected_nodes[j] == 1:
                num_cuts += 1
                node_0, node_1 = np.array([]), np.array([])
                flag0, flag1 = 0, 0
                it, jt = 0, 0
                for i in range(len(super_nodes)):
                    if con_id_i in super_nodes[i]:
                        node_0 = super_nodes[i]
                        it = i
                        flag0 = 1
                    elif con_id_j in super_nodes[i]:
                        node_1 = super_nodes[i]
                        jt = i
                        flag1 = 1
                    if flag0 == 1 and flag1 == 1:
                        break
                node_0 += node_1

            else:
                for n in super_nodes:
                    if con_id_i in n:
                        # the super node already exist
                        n.append(con_id_j)
                        flag = 1
                        num_cuts += 1
                        break
                    elif con_id_j in n:
                        n.append(con_id_i)
                        flag = 1
                        num_cuts += 1
                        break

                if flag == 0:
                    num_cuts += 1
                    nodes = [con_id_i, con_id_j]
                    super_nodes.append(nodes)

                selected_nodes[i] = 1
                selected_nodes[j] = 1
                # print(super_nodes)
                # print(selected_nodes)

        if np.count_nonzero(selected_nodes) != shape:
            unselected_nodes = np.where((selected_nodes == 0))[0].tolist()
            for us in unselected_nodes:
                # super_nodes.append(np.array([self.vector_id_list[us]]))
                super_nodes.append([self.vector_id_list[us]])

        return self.vector_id_list, super_nodes

    def min_cut1(self, containers, os) -> None:
        """ cut the application into two partitions
        """
        container_clusters = np.array([0, 0, 0, 0])         # the final container clusters
        shape = self.communication.shape[0]
        selected_nodes = np.zeros(shape)
        backlist = []                   # if in this list, it cannot be selected to cluster
        num_cuts = 0                    # count the number of cuts
        mapping = {}                    # record which container id in which partition
        
        super_nodes = []
        communication = copy.deepcopy(self.communication)
        while num_cuts < 4 and len(backlist) < len(self.vector_id_list) and \
                len(self.vector_id_list) > 1:
            if np.count_nonzero(selected_nodes) == shape:
                break

            index = np.unravel_index(communication.argmax(), communication.shape)
            i, j = index[0], index[1]

            con_id_i, con_id_j = self.vector_id_list[i], self.vector_id_list[j]

            if (i not in backlist) and (j not in backlist):
                if con_id_i in mapping.keys():
                    if mapping[con_id_i][0] + containers.iloc[con_id_j]["cpu"] > 27733.333333333332 or \
                        mapping[con_id_i][1] + containers.iloc[con_id_j]["memory"] > 21000:
                        communication[i][j] = -1 * communication[i][j]
                        backlist.append(i)
                        backlist.append(j)
                        continue
                elif con_id_j in mapping.keys():
                    if mapping[con_id_j][0] + containers.iloc[con_id_i]["cpu"] > 27733.333333333332 or \
                        mapping[con_id_j][1] + containers.iloc[con_id_i]["memory"] > 21000:
                        communication[i][j] = -1 * communication[i][j]
                        backlist.append(i)
                        backlist.append(j)
                        continue
                elif con_id_i in mapping.keys() and con_id_j in mapping.keys():
                    if mapping[con_id_j][0] + mapping[con_id_i][0] > 27733.333333333332 or \
                        mapping[con_id_j][1] + mapping[con_id_i][1] > 21000:
                        communication[i][j] = -1 * communication[i][j]
                        backlist.append(i)
                        backlist.append(j)
                        continue
                elif containers.iloc[con_id_i]["cpu"] + containers.iloc[con_id_j]["cpu"] > 27733.333333333332 or \
                    containers.iloc[con_id_i]["memory"] + containers.iloc[con_id_j]["memory"] > 21000:
                    communication[i][j] = -1 * communication[i][j]
                    backlist.append(i)
                    backlist.append(j)
                    continue
            else:
                backlist.append(i)
                backlist.append(j)
                continue
            
            communication[:, i] += communication[:, j]
            communication[i, i] = 0
            communication[j, j] = 0
            communication[:, j] = 0
            communication[i, :] += communication[j, :]
            communication[i, i] = 0
            communication[j, j] = 0
            communication[j, :] = 0
            flag = 0
            if selected_nodes[i] == 1 and selected_nodes[j] == 1:
                num_cuts += 1
                node_0, node_1 = np.array([]), np.array([])
                flag0, flag1 = 0, 0
                it, jt = 0, 0
                for i in range(len(super_nodes)):
                    if con_id_i in super_nodes[i]:
                        node_0 = super_nodes[i]
                        it = i
                        flag0 = 1
                    elif con_id_j in super_nodes[i]:
                        node_1 = super_nodes[i]
                        jt = i
                        flag1 = 1
                    if flag0 == 1 and flag1 == 1:
                        break
                node_0 = np.append(node_0, node_1)
                super_nodes[it] = node_0
                del super_nodes[jt]

                for n_1 in node_1:
                    mapping[node_0[0]][0] += containers.iloc[n_1]["cpu"]
                    mapping[node_0[0]][1] += containers.iloc[n_1]["memory"]
                
            else:
                for n in super_nodes:
                    if con_id_i in n:
                        # the super node already exist
                        n = np.append(n, con_id_j)
                        flag = 1
                        num_cuts += 1
                        mapping[con_id_i][0] += containers.iloc[con_id_j]["cpu"]
                        mapping[con_id_i][1] += containers.iloc[con_id_j]["memory"]
                        mapping[con_id_j] = mapping[con_id_i]
                        break
                    elif con_id_j in n:
                        n = np.append(n, con_id_i)
                        flag = 1
                        num_cuts += 1
                        mapping[con_id_j][0] += containers.iloc[con_id_i]["cpu"]
                        mapping[con_id_j][1] += containers.iloc[con_id_i]["memory"]
                        mapping[con_id_i] = mapping[con_id_j]
                        break

                if flag == 0:
                    num_cuts += 1
                    nodes = np.array([con_id_i, con_id_j])
                    super_nodes.append(nodes)
                    cpu = containers.iloc[con_id_i]["cpu"] + containers.iloc[con_id_j]["cpu"]
                    memory = containers.iloc[con_id_i]["memory"] + containers.iloc[con_id_j]["memory"]
                    timestamp = containers.iloc[con_id_i]["timestamp"]
                    os_id = os.iloc[con_id_i]["os-id"]
                    cluster = np.array([cpu, memory, timestamp, os_id])
                    mapping[con_id_i] = cluster
                    mapping[con_id_j] = cluster
                    container_clusters = np.row_stack((container_clusters, cluster))

                selected_nodes[i] = 1
                selected_nodes[j] = 1

        if np.count_nonzero(selected_nodes) != shape:
            unselected_nodes = np.where((selected_nodes == 0))[0].tolist()
            for us in unselected_nodes:
                super_nodes.append(np.array([self.vector_id_list[us]]))
                cluster = np.array([containers.iloc[self.vector_id_list[us]]["cpu"], containers.iloc[self.vector_id_list[us]]["memory"], containers.iloc[self.vector_id_list[us]]["timestamp"], os.iloc[self.vector_id_list[us]]["os-id"]])
                container_clusters = np.row_stack((container_clusters, cluster))
                mapping[self.vector_id_list[us]] = cluster

        self.communication = np.absolute(self.communication)        # take the absolute of the communication
        container_clusters = np.delete(container_clusters, 0, axis=0)
        container_clusters = pd.DataFrame(container_clusters, columns=["cpu", "memory", "timestamp", "os-id"])
        # print(super_nodes)
        # print(container_clusters)
        return container_clusters, super_nodes

    # # def generate_container_cluster(self, containers, os) -> DataFrame:
    # #     """ cluster containers into serveral partitions based on the super nodes
    # #     """
    # #     container_clusters = []
    # #     columns = ["cpu", "memory", "timestamp", "os-id"]
    # #     for n in self.super_nodes:
    # #         cluster = []
    # #         for i in n:
    # #             con_info = [containers.iloc[i]["cpu"], containers.iloc[i]["memory"],
    # #                         containers.iloc[i]["timestamp"], os.iloc[i]["os-id"]]
    # #             cluster.append(con_info)
            
    # #         cluster = pd.DataFrame(cluster, columns=columns)
    # #         groups = cluster.groupby("os-id")
    # #         for name, group in groups:
    # #             cluster_c = []
    # #             cpu = 0
    # #             memory = 0
    # #             timestamp = group.iloc[0, 2]
    # #             os_id = group.iloc[0, 3]
    # #             num = 0           # used to identify whether it is the last line
    # #             for row in group.iterrows():
    # #                 num += 1
    # #                 if cpu + row[1]["cpu"] <= 27733.333333333332 \
    # #                         and memory + row[1]["memory"] <= 21000:
    # #                     # if cluster less than the largest VM
    # #                     cpu += row[1]["cpu"]
    # #                     memory += row[1]["memory"]
    # #                 else:
    # #                     cluster_c.append(cpu)
    # #                     cluster_c.append(memory)
    # #                     cluster_c.append(timestamp)
    # #                     cluster_c.append(os_id)
    # #                     container_clusters.append(cluster_c)
    # #                     cluster_c = []
    # #                     cpu = 0
    # #                     memory = 0
    # #                 if num == group.shape[0]:
    # #                     cluster_c.append(cpu)
    #                     cluster_c.append(memory)
    #                     cluster_c.append(timestamp)
    #                     cluster_c.append(os_id)
    #                     container_clusters.append(cluster_c)

    #     container_clusters = pd.DataFrame(container_clusters, columns=columns)

    #     return container_clusters, self.vector_id_list


    # def sort_services(self) -> list:
    #     print("before = ", self.vector_id_list)
    #     self.vector_id_list.sort(key=functools.cmp_to_key(self.cmp_services))
    #     print("after = ", self.vector_id_list)

    #     return self.vector_id_list
    
    # def sort_test_services(self) -> list:
    #     # self.vector_id_list.sort(key=functools.cmp_to_key(self.cmp_test_services))
    #     # dataset = config.RUNNING_PARAMS['dataset']
    #     # os_dataset = config.RUNNING_PARAMS['OS-dataset']
    #     # container_data_dir = DATASET_DIR_LOOKUP[dataset] + '/' + os_dataset + '/Test/containerData/testCase{:}.csv'.format(0)

    #     # containers = pd.read_csv(container_data_dir, header=None, names=COLUMN_NAMES_LOOKUP['container-data'])

    #     # print("before = ", self.vector_id_list)
    #     self.vector_id_list.sort(key=functools.cmp_to_key(self.cmp_test_services))
    #     # print("after = ", self.vector_id_list)

    #     return self.vector_id_list

    # def cmp_services(self, a: int, b: int) -> int:
    #     """ Sort the services list by normalized resource
    #     """
    #     dataset = config.RUNNING_PARAMS['dataset']
    #     os_dataset = config.RUNNING_PARAMS['OS-dataset']
    #     container_data_dir = DATASET_DIR_LOOKUP[dataset] + '/' + os_dataset + '/containerData/testCase{:}.csv'.format(self.test_num)

    #     containers = pd.read_csv(container_data_dir, header=None, names=COLUMN_NAMES_LOOKUP['container-data'])

    #     nra = containers.loc[a]["cpu"] / containers["cpu"].max() * containers.loc[a]["memory"] / containers["memory"].max()
    #     nrb = containers.loc[b]["cpu"] / containers["cpu"].max() * containers.loc[b]["memory"] / containers["memory"].max()
        
    #     if nra > nrb:
    #         return -1
    #     elif nra < nrb:
    #         return 1
    #     else:
    #         return 0

    # def cmp_test_services(self, a: int, b: int) -> int:
    #     """ Sort the services list by normalized resource
    #     """
        
    #     nra = self.test_containers.loc[a]["cpu"] / self.test_containers["cpu"].max() * self.test_containers.loc[a]["memory"] / self.test_containers["memory"].max()
    #     nrb = self.test_containers.loc[b]["cpu"] / self.test_containers["cpu"].max() * self.test_containers.loc[b]["memory"] / self.test_containers["memory"].max()
        
    #     if nra > nrb:
    #         return -1
    #     elif nra < nrb:
    #         return 1
    #     else:
    #         return 0

    # # def test_quick_sort(self, containers, lists, i, j):
    # #     if i >= j:
    # #         return list
    # #     low = i
    # #     high = j

    # #     pivot = containers.loc[i]["cpu"] / containers["cpu"].max() * containers.loc[i]["memory"] / containers["memory"].max()
    # #     pivot_index = lists[i]

    # #     while i < j:
    # #         while i < j and containers.loc[j]["cpu"] / containers["cpu"].max() * containers.loc[j]["memory"] / containers["memory"].max() >= pivot:
    # #             j -= 1
    # #         lists[i] = lists[j]
    # #         while i < j and containers.loc[i]["cpu"] / containers["cpu"].max() * containers.loc[i]["memory"] / containers["memory"].max() <=pivot:
    # #             i += 1
    # #         lists[j] = lists[i]
    # #     lists[j] = pivot_index
    # #     self.test_quick_sort(containers, lists, low, i-1)
    # #     self.test_quick_sort(containers, lists, i+1, high)

    # #     return lists
