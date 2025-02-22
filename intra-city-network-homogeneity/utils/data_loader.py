import os
import re
import json
import numpy as np
import networkx as nx


class DataLoader():
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data = {}
        self.ids = []
        self.cities = set()
        self.samples = {}
        self.load_cities()

    def load_cities(self):
        files = os.listdir(self.data_dir)
        for file in files:
            name = file[:-5]
            number = re.findall('[0-9]+', name)[0]
            city, attr = name.split(number)
            if city not in self.cities:
                self.cities.add(city)
            if city not in self.samples:
                self.samples[city] = set()
            self.samples[city].add(number)
        self.cities = list(self.cities)
        for city in self.samples:
            self.samples[city] = list(self.samples[city])

    def load_all_datas(self):
        for city in self.cities:
            self.load_dir_datas(city)

    def initialize(self):
        self.data = {}
        self.ids = []

    def load_one_sample(self, sample):
        name = sample[:-5]
        number = re.findall('[0-9]+', name)[0]
        city, attr = name.split(number)
        id1 = city + '_' + number + '_1'
        id2 = city + '_' + number + '_2'
        if city not in self.data:
            self.data[city] = {}
        if id1 not in self.data[city]:
            self.data[city][id1] = {'id': id1, 'nodes': [], 'source_edges': [], 'target_edges': []}
            self.data[city][id2] = {'id': id2, 'nodes': [], 'source_edges': [], 'target_edges': []}
            self.ids += [id1, id2]
        data = json.load(open(self.data_dir + sample, 'r'))
        if attr == 'nodes':
            self.data[city][id1]['nodes'] = data
            self.data[city][id2]['nodes'] = data
        elif attr == 'edges':
            for edge in data:
                if edge['inSample1'] == 1:
                    self.data[city][id1]['source_edges'].append({
                        'start': edge['start'],
                        'end': edge['end'],
                    })
                if edge['inSample1'] == 0:
                    self.data[city][id1]['target_edges'].append({
                        'start': edge['start'],
                        'end': edge['end'],
                    })
                if edge['inSample2'] == 1:
                    self.data[city][id2]['source_edges'].append({
                        'start': edge['start'],
                        'end': edge['end'],
                    })
                if edge['inSample2'] == 0:
                    self.data[city][id2]['target_edges'].append({
                        'start': edge['start'],
                        'end': edge['end'],
                    })

    def load_dir_datas(self, cityname):
        files = os.listdir(self.data_dir)
        for file in files:
            name = file[:-5]
            number = re.findall('[0-9]+', name)[0]
            city, attr = name.split(number)
            if city != cityname:
                continue
            id1 = city + '_' + number + '_1'
            id2 = city + '_' + number + '_2'
            if city not in self.data:
                self.data[city] = {}
            if id1 not in self.data[city]:
                self.data[city][id1] = {'id': id1, 'nodes': [], 'source_edges': [], 'target_edges': []}
                self.data[city][id2] = {'id': id2, 'nodes': [], 'source_edges': [], 'target_edges': []}
                self.ids += [id1, id2]

            data = json.load(open(self.data_dir + file, 'r'))
            if attr == 'nodes':
                self.data[city][id1]['nodes'] = data
                self.data[city][id2]['nodes'] = data
            elif attr == 'edges':
                for edge in data:
                    if edge['inSample1'] == 1:
                        self.data[city][id1]['source_edges'].append({
                            'start': edge['start'],
                            'end': edge['end'],
                        })
                    if edge['inSample1'] == 0:
                        self.data[city][id1]['target_edges'].append({
                            'start': edge['start'],
                            'end': edge['end'],
                        })
                    if edge['inSample2'] == 1:
                        self.data[city][id2]['source_edges'].append({
                            'start': edge['start'],
                            'end': edge['end'],
                        })
                    if edge['inSample2'] == 0:
                        self.data[city][id2]['target_edges'].append({
                            'start': edge['start'],
                            'end': edge['end'],
                        })

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        if type(index) == int:
            index = self.ids[index]
        city, number, idx = index.split('_')
        return self.data[city][index]

    @staticmethod
    def build_graph(nodes, edges):
        ids = [str(n['osmid']) for n in nodes]
        edges = [(str(e['start']), str(e['end'])) for e in edges] + [(str(e['end']), str(e['start'])) for e in edges]
        graph = nx.DiGraph(np.array([[0] * len(nodes)] * len(nodes)))

        mapping = {i: ids[i] for i in range(len(ids))}
        graph = nx.relabel_nodes(graph, mapping)
        graph.add_edges_from(edges)
        return graph

    def build_source_graph(self, index):
        data = self[index]
        return DataLoader.build_graph(data['nodes'], data['source_edges'])

    def build_full_graph(self, index):
        data = self[index]
        return DataLoader.build_graph(data['nodes'], data['source_edges'] + data['target_edges'])


if __name__ == "__main__":
    train = DataLoader('E:/python-workspace/CityRoadPrediction/data_20200610/train/')
    test = DataLoader('E:/python-workspace/CityRoadPrediction/data_20200610/test/')
    print('city num:', len(train.data))
    for k in train.data:
        if k not in test.data:
            print(k, len(train.data[k]))
        else:
            print(k, len(train.data[k]), len(test.data[k]))

