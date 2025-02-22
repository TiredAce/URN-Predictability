import numpy as np
import json
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import networkx as nx
import math
import glob
from sklearn.cluster import KMeans
import os
from scipy.spatial.distance import cdist
from sklearn.metrics import f1_score
from sklearn import linear_model
import scipy

font1 = {'family' : 'Arial',
'weight' : 'normal',
'size'   : 23,
}
font2 = {'family' : 'Arial',
'weight' : 'normal',
'size'   : 30,
}

def nodes_to_list(nodes):
    new_nodes = []
    for n in nodes:
        new_nodes.append([n['osmid'],n['lon'],n['lat']])
    return new_nodes

def edges_to_dict(edges, sample=1):
    old_edges = {}
    for e in edges:
        if sample == 1:
            if e['start'] not in old_edges:
                old_edges[e['start']] = []
            old_edges[e['start']].append(e['end'])
        if sample == 2:
            if e['start'] not in old_edges:
                old_edges[e['start']] = []
            old_edges[e['start']].append(e['end'])
    return old_edges

def load_graph(file_name, sample=1):
    nodes = json.load(open('../codeJiaweiXue_2020715_dataCollection/train/'+file_name+'nodes.json', 'r'))
    edges = json.load(open('../codeJiaweiXue_2020715_dataCollection/train/'+file_name+'edges.json', 'r'))
    old_edges = edges_to_dict(edges, sample=sample)
    return nodes, old_edges

def visualization(nodeInfor, predictEdges, oldEdges, newEdges, city_name, cluster, rank, title):
    # step0: get the information
    nodeId = [nodeInfor[i][0] for i in range(len(nodeInfor))]
    longitude = [nodeInfor[i][1] for i in range(len(nodeInfor))]
    latitude = [nodeInfor[i][2] for i in range(len(nodeInfor))]

    # step1: generate the graph
    n = len(nodeId)
    A1 = np.array([[0] * n] * n)
    Graph1 = nx.Graph(A1)

    # step 2: label
    column = [str(nodeId[i]) for i in range(n)]
    mapping = {0: str(nodeId[0])}
    for i in range(0, len(column) - 1):
        mapping.setdefault(i + 1, column[i + 1])
    Graph1 = nx.relabel_nodes(Graph1, mapping)

    # step3: geolocation
    POS = list()
    for i in range(0, n):
        POS.append((float(longitude[i]), float(latitude[i])))
    for i in range(0, n):
        Graph1.nodes[column[i]]['pos'] = POS[i]

    num = 0
    # step 4: add edge
    for start in oldEdges:
        for end in oldEdges[start]:
            num = num + 1
            Graph1.add_edge(str(start), str(end), color='black', weight=1)
    # print('old num', num)
    for start in newEdges:
        for end in newEdges[start]:
            if (not (start in predictEdges and end in predictEdges[start])) and \
                    (not (end in predictEdges and start in predictEdges[end])):
                Graph1.add_edge(str(start), str(end), color='blue', weight=2)
    for start in predictEdges:
        for end in predictEdges[start]:
            if (start in newEdges and end in newEdges[start]) or \
                    (end in newEdges and start in newEdges[end]):
                Graph1.add_edge(str(start), str(end), color='green', weight=5)
            else:
                Graph1.add_edge(str(start), str(end), color='red', weight=2)

    edges = Graph1.edges()
    colors = [Graph1[u][v]['color'] for u, v in edges]
    weights = [Graph1[u][v]['weight'] for u, v in edges]
    # print(nx.cycle_basis(Graph1))
    plt.figure(1, figsize=(6, 6))
    if title:
        if rank>=0:
            plt.title('city: {} cluster: {} rank: {}'.format(city_name, cluster, rank))
        else:
            plt.title('city: {} cluster: {}'.format(city_name, cluster))

    nx.draw(Graph1, nx.get_node_attributes(Graph1, 'pos'), edge_color=colors, width=weights, node_size=10)#, with_labels = True)
    plt.show()
    if not os.path.exists('figures/'+str(cluster)):
        os.mkdir('figures/'+str(cluster)+'/')
    # if title:
    #     plt.savefig('figures/{}/cluster_{}_'.format(cluster, cluster)+city_name+'.png')
    # else:
    #     plt.savefig('figures/{}/rank_{}_cluster_{}_'.format(cluster, rank, cluster) + city_name + '.png')
    # plt.clf()

def visualize(city_name, cluster, rank = -1, title = True):
    sample = 1
    nodes, old_edges = load_graph(city_name, sample)
    visualization(nodes_to_list(nodes), dict(), old_edges, dict(), city_name, cluster, rank = rank, title = title)

def pca_visualize(k):
    with open('results/training_set_index.txt') as json_file:
        city_index = json.load(json_file)
    data = np.zeros((len(city_index), 11))
    num_2_cityname = {}
    for city_num, city in enumerate(city_index):
        num_2_cityname[city_num] = city
        for idx_num, attribute in enumerate(city_index[city]):
            data[city_num, idx_num] = city_index[city][attribute]
    print('training data shape: ', data.shape)

    data_mean = np.mean(data, axis=0, keepdims=True)
    data_std = np.std(data, axis=0, keepdims=True)
    data = (data - data_mean)/data_std

    k = k

    kmeanModel = KMeans(n_clusters=k, random_state = 1)
    kmeanModel.fit(data)

    # print(kmeanModel.labels_== 1)
    pca = PCA(n_components=6)
    newData = pca.fit_transform(data)
    print((np.transpose(pca.components_)).shape)
    print('PCA component', (np.transpose(pca.components_)))
    print('explained variance', pca.explained_variance_)
    print('explained variance ratio', pca.explained_variance_ratio_)

    # the index of cluster is ordered by the value of PCA1
    change_order = True
    if change_order:
        SDNi = np.zeros(k)
        change_order_mapping = {}
        for i in range(k):
            SDNi[i] = np.mean(newData[kmeanModel.labels_== i][:,0])
        argsorted_SDNi = np.argsort(SDNi)
        for i in range(k):
            change_order_mapping[i] = np.where(argsorted_SDNi==i)[0][0]
        for i in range(len(kmeanModel.labels_)):
            kmeanModel.labels_[i] = change_order_mapping[kmeanModel.labels_[i]]

    cluster_centers_ = np.zeros((k, data.shape[1]))
    for i in range(k):
        cluster_centers_[i,:] = np.mean(data[kmeanModel.labels_== i], axis = 0, keepdims=True)

    pair_distance = cdist(data, cluster_centers_, 'euclidean')

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))#, dpi = 1200)
    plt.plot(newData[kmeanModel.labels_== 0][:,0], newData[kmeanModel.labels_== 0][:,1], 'o', markersize=3, label="Type 1")
    plt.plot(newData[kmeanModel.labels_== 1][:,0], newData[kmeanModel.labels_== 1][:,1], 'o', markersize=3, label="Type 2")
    plt.plot(newData[kmeanModel.labels_== 2][:,0], newData[kmeanModel.labels_== 2][:,1], 'o', markersize=3, label="Type 3")
    plt.plot(newData[kmeanModel.labels_== 3][:,0], newData[kmeanModel.labels_== 3][:,1], 'o', markersize=3, label="Type 4")

    plt.yticks(fontsize=22)
    plt.xticks(fontsize=22)
    plt.xlabel('Dimension 1', font1)
    plt.ylabel('Dimension 2', font1)
    plt.legend(loc="best", fontsize=21.3, markerscale=3., labelspacing = 0.2, borderpad = 0.25)
    plt.tight_layout()
    plt.savefig('figures/pca.png', bbox_inches='tight')
    plt.show()

def city_ratio(k):
    with open('results/training_set_index.txt') as json_file:
        city_index = json.load(json_file)
    data = np.zeros((len(city_index), 11))
    num_2_cityname = {}
    for city_num, city in enumerate(city_index):
        num_2_cityname[city_num] = city
        for idx_num, attribute in enumerate(city_index[city]):
            data[city_num, idx_num] = city_index[city][attribute]
    print('training data shape: ', data.shape)

    data_mean = np.mean(data, axis=0, keepdims=True)
    data_std = np.std(data, axis=0, keepdims=True)
    data = (data - data_mean)/data_std

    with open('results/test_set_index.txt') as json_file:
        city_index = json.load(json_file)
    test_data = np.zeros((len(city_index), 11))
    test_num_2_cityname = {}
    for city_num, city in enumerate(city_index):
        test_num_2_cityname[city_num] = city
        for idx_num, attribute in enumerate(city_index[city]):
            test_data[city_num, idx_num] = city_index[city][attribute]
    print('testing data shape: ', test_data.shape)

    test_data = (test_data - data_mean) / data_std

    k = k

    kmeanModel = KMeans(n_clusters=k, random_state=1)
    kmeanModel.fit(data)

    pca = PCA(n_components=6)
    newData = pca.fit_transform(data)
    test_newData = np.matmul(test_data, np.transpose(pca.components_))

    ### the index of cluster is ordered by the increasing order of PCA1
    change_order = True
    if change_order:
        SDNi = np.zeros(k)
        change_order_mapping = {}
        for i in range(k):
            SDNi[i] = np.mean(newData[kmeanModel.labels_ == i][:, 0])
        argsorted_SDNi = np.argsort(SDNi)
        for i in range(k):
            change_order_mapping[i] = np.where(argsorted_SDNi == i)[0][0]
        for i in range(len(kmeanModel.labels_)):
            kmeanModel.labels_[i] = change_order_mapping[kmeanModel.labels_[i]]

    ### get the label for the testing data
    cluster_centers_ = np.zeros((k, data.shape[1]))
    for i in range(k):
        cluster_centers_[i, :] = np.mean(data[kmeanModel.labels_ == i], axis=0, keepdims=True)

    pair_distance = cdist(test_data, cluster_centers_, 'euclidean')
    test_data_assign_label = np.argmin(pair_distance, axis=1)

    ## read test data f1 value
    test_result = json.load(open('results/f1_score_test_result.json', 'r'))
    test_result_ = {}
    for city in test_result:
        for city_name in test_result[city]:
            new_city_name = city_name.split('_')[0] + city_name.split('_')[1] + '_' + city_name.split('_')[2]
            test_result_[new_city_name] = test_result[city][city_name]

    results_city = {}

    cityName = ['Chicago', 'New york', 'Los angeles', 'Tokyo', 'Berlin', 'Phoenix', 'Paris', 'London', 'Hongkong', 'Singapore']
    for city in cityName:
        results_city[city] = []

    for idx in range(test_data_assign_label.shape[0]):
        for city in cityName:
            if city in test_num_2_cityname[idx]:
                results_city[city].append(test_result_[test_num_2_cityname[idx] + '_1'])
                results_city[city].append(test_result_[test_num_2_cityname[idx] + '_2'])

    # we sort the order of visualiztion by its median f1 value
    def sortFunc(e):
        return np.median(results_city[e])
    cityName.sort(key=sortFunc, reverse=True)

    count = {}

    for city in cityName:
        if city not in count:
            count[city] = []
    for idx, label in enumerate(test_data_assign_label):
        for city in cityName:
            if city in test_num_2_cityname[idx]:
                count[city].append(label)

    ratio = {}
    for city in count:
        if city not in ratio:
            ratio[city] = np.zeros(k)
        for i in range(k):
            ratio[city][i] = np.sum(np.array(count[city])==i)
        ratio[city] = ratio[city]/np.sum(ratio[city])

    category_names = ['T ' + str(i + 1) for i in range(6)]

    results = {}
    for city in count:
        results[city] = ratio[city]
    # step 2: figure, label
    labels = cityName#list(results.keys())
    data = np.array(list(results.values()))

    ## this is used to control the length of each bar
    data_visualize = np.array(list(results.values()))+0.05
    data_visualize = data_visualize/np.sum(data_visualize, axis=1, keepdims=True)
    data_cum = data_visualize.cumsum(axis=1)

    # category_colors = plt.get_cmap('RdYlGn')(np.linspace(0.1, 1.0, data.shape[1]))
    category_colors = ['pink', 'lightblue', 'lightgreen', 'lightyellow', 'lightsalmon']
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 7))#, dpi=1200)

    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        if i>=k:
            break
        widths = data_visualize[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, width = widths, left=starts, height=0.85, label=colname, color=color, edgecolor="black")
        xcenters = starts + widths / 2
        text_color = 'black'
        for y, (x, c) in enumerate(zip(xcenters, data[:, i])):
            ax.text(x, y, '{}%'.format(int(round(c*100))), ha='center', va='center', color=text_color, fontsize=12)
    ax.legend(ncol=4, bbox_to_anchor=(0, 1), loc='lower left', fontsize=18, labelspacing = 0.1, borderpad = 0.20)
    plt.yticks(fontsize=18, rotation=45)
    plt.tight_layout()
    plt.savefig('figures/city_ratio.png', bbox_inches='tight')
    plt.show()

def f1_vs_network_type(k):
    with open('results/training_set_index.txt') as json_file:
        city_index = json.load(json_file)
    data = np.zeros((len(city_index), 11))
    num_2_cityname = {}
    for city_num, city in enumerate(city_index):
        num_2_cityname[city_num] = city
        for idx_num, attribute in enumerate(city_index[city]):
            data[city_num, idx_num] = city_index[city][attribute]
    print('training data shape: ', data.shape)

    data_mean = np.mean(data, axis=0, keepdims=True)
    data_std = np.std(data, axis=0, keepdims=True)
    data = (data - data_mean)/data_std

    with open('results/test_set_index.txt') as json_file:
        city_index = json.load(json_file)
    test_data = np.zeros((len(city_index), 11))
    test_num_2_cityname = {}
    for city_num, city in enumerate(city_index):
        test_num_2_cityname[city_num] = city
        for idx_num, attribute in enumerate(city_index[city]):
            test_data[city_num, idx_num] = city_index[city][attribute]
    print('test data shape: ', test_data.shape)

    test_data = (test_data - data_mean)/data_std

    k = k

    kmeanModel = KMeans(n_clusters=k, random_state = 1)
    kmeanModel.fit(data)

    pca = PCA(n_components=6)
    newData = pca.fit_transform(data)
    test_newData = np.matmul(test_data,np.transpose(pca.components_))

    change_order = True
    if change_order:
        SDNi = np.zeros(k)
        change_order_mapping = {}
        for i in range(k):
            SDNi[i] = np.mean(newData[kmeanModel.labels_== i][:,0])
        argsorted_SDNi = np.argsort(SDNi)
        for i in range(k):
            change_order_mapping[i] = np.where(argsorted_SDNi==i)[0][0]
        for i in range(len(kmeanModel.labels_)):
            kmeanModel.labels_[i] = change_order_mapping[kmeanModel.labels_[i]]

    cluster_centers_ = np.zeros((k, data.shape[1]))
    for i in range(k):
        cluster_centers_[i,:] = np.mean(data[kmeanModel.labels_== i], axis = 0, keepdims=True)

    pair_distance = cdist(test_data, cluster_centers_, 'euclidean')
    test_data_assign_label = np.argmin(pair_distance, axis = 1)

    ## read test data f1 value
    test_result = json.load(open('results/f1_score_test_result.json', 'r'))
    test_result_ = {}
    for city in test_result:
        for city_name in test_result[city]:
            new_city_name = city_name.split('_')[0]+city_name.split('_')[1]+'_'+city_name.split('_')[2]
            test_result_[new_city_name] = test_result[city][city_name]

    results_road_type = {}
    for i in range(k):
        results_road_type[i] = []
    for idx in range(test_data_assign_label.shape[0]):
        if test_num_2_cityname[idx]+'_1' in test_result_:
            results_road_type[test_data_assign_label[idx]].append(test_result_[test_num_2_cityname[idx] + '_1'])
            results_road_type[test_data_assign_label[idx]].append(test_result_[test_num_2_cityname[idx] + '_2'])

    all_data = []
    for i in range(k):
        all_data.append(results_road_type[i])
    cityName = ['Type 1', 'Type 2', 'Type 3', 'Type 4']
    # step 2: figure
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))#, dpi=1200)
    bplot1 = ax1.boxplot(all_data,
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
                         labels=cityName)  # will be used to label x-ticks
    # ax1.set_title("F1 scores for different road network types", font1, fontsize=20)
    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.yticks(fontsize=22)
    plt.xticks(fontsize=22, rotation=0)
    # step 3: color
    colors = ['pink', 'lightblue', 'lightgreen', 'lightyellow', 'lightsalmon']
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
    # step 4: grid
    ax1.yaxis.grid(True)
    plt.ylabel('F1 score', font1, fontsize=22)
    plt.tight_layout()
    plt.savefig('figures/citytype_vs_f1.png', bbox_inches='tight')
    plt.show()

def f1_vs_city():
    with open('results/test_set_index.txt') as json_file:
        city_index = json.load(json_file)
    test_data = np.zeros((len(city_index), 11))
    test_num_2_cityname = {}
    for city_num, city in enumerate(city_index):
        test_num_2_cityname[city_num] = city
        for idx_num, attribute in enumerate(city_index[city]):
            test_data[city_num, idx_num] = city_index[city][attribute]
    print('test data shape: ', test_data.shape)

    ## read test data f1 value
    test_result = json.load(open('results/f1_score_test_result.json', 'r'))
    test_result_ = {}
    for city in test_result:
        for city_name in test_result[city]:
            new_city_name = city_name.split('_')[0]+city_name.split('_')[1]+'_'+city_name.split('_')[2]
            test_result_[new_city_name] = test_result[city][city_name]
    print(len(test_result_))
    results_city = {}
    cityName = ['Chicago', 'New york', 'Los angeles', 'Tokyo', 'Berlin', 'Phoenix', 'Paris', 'London', 'Hongkong', 'Singapore']
    for city in cityName:
        results_city[city] = []

    for idx in range(len(city_index)):
        for city in cityName:
            if city in test_num_2_cityname[idx]:
                print(idx, test_num_2_cityname[idx], test_result_[test_num_2_cityname[idx]+'_1'], test_result_[test_num_2_cityname[idx]+'_2'])
                results_city[city].append(test_result_[test_num_2_cityname[idx] + '_1'])
                results_city[city].append(test_result_[test_num_2_cityname[idx] + '_2'])

    for city in cityName:
        print(city, len(results_city[city])/2)

    ####
    all_data = []
    for city in cityName:
        all_data.append(results_city[city])

    # # step 2: figure
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))#, dpi=1200)
    bplot1 = ax1.boxplot(all_data,
                         vert=True,  # vertical box alignment
                         patch_artist=True,  # fill with color
                         labels=cityName)  # will be used to label x-ticks
    # ax1.set_title("F1 scores for different cities", font1, fontsize=20)
    ax1.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.yticks(fontsize=22)
    plt.xticks(fontsize=16, rotation=52)
    # step 3: color
    colors = ['pink', 'lightblue', 'lightgreen', 'lightyellow', 'lightsalmon', 'pink', 'lightblue', 'lightgreen',
              'lightyellow', 'lightsalmon', ]
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
    # step 4: grid
    plt.ylabel('F1 score', font1, fontsize=22)
    ax1.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig('figures/city_vs_f1.png', bbox_inches='tight')
    plt.show()

def f1_vs_PCA1():
    with open('results/training_set_index.txt') as json_file:
        city_index = json.load(json_file)
    data = np.zeros((len(city_index), 11))
    num_2_cityname = {}
    for city_num, city in enumerate(city_index):
        num_2_cityname[city_num] = city
        for idx_num, attribute in enumerate(city_index[city]):
            data[city_num, idx_num] = city_index[city][attribute]
    print('data shape: ', data.shape)

    data_mean = np.mean(data, axis=0, keepdims=True)
    data_std = np.std(data, axis=0, keepdims=True)
    data = (data - data_mean) / data_std

    with open('results/test_set_index.txt') as json_file:
        city_index = json.load(json_file)
    test_data = np.zeros((len(city_index), 11))
    test_num_2_cityname = {}
    for city_num, city in enumerate(city_index):
        test_num_2_cityname[city_num] = city
        for idx_num, attribute in enumerate(city_index[city]):
            test_data[city_num, idx_num] = city_index[city][attribute]
    print('test data shape: ', test_data.shape)

    test_data = (test_data - data_mean) / data_std

    pca = PCA(n_components=6)
    newData = pca.fit_transform(data)

    test_newData = np.matmul(test_data, np.transpose(pca.components_))

    test_result = json.load(open('results/f1_score_test_result.json', 'r'))
    test_result_ = {}
    for city in test_result:
        for city_name in test_result[city]:
            new_city_name = city_name.split('_')[0] + city_name.split('_')[1] + '_' + city_name.split('_')[2]
            test_result_[new_city_name] = test_result[city][city_name]

    f1_score = np.zeros(test_newData.shape[0])
    for i in range(test_newData.shape[0]):
        f1_score[i] = (test_result_[test_num_2_cityname[i] + '_1'] + test_result_[test_num_2_cityname[i] + '_2']) / 2

    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(7, 7))#, dpi = 1200)
    plt.scatter(test_newData[:, 0], f1_score, s=3, label='Road network')

    # Create linear regression object
    regr = linear_model.LinearRegression()
    # Train the model using the training sets
    regr.fit(test_newData[:, 0].reshape(-1, 1), np.reshape(f1_score, (-1, 1)))

    test_X = np.arange(test_newData[:, 0].min(), test_newData[:, 0].max(), 0.05).reshape(-1, 1)
    # Make predictions using the testing set
    test_y_pred = regr.predict(test_X)

    plt.plot(test_X, test_y_pred, linewidth=3, label='Linear regression', color='r')

    pca1_f1 = np.column_stack((test_newData[:, 0]/5, f1_score))

    pts = np.array([[-4.,0.9], [4.0,0.5], [4.0,0.1]])
    color = plt.rcParams['axes.prop_cycle'].by_key()['color']
    color.remove(color[3])
    color.remove(color[0])
    pts[:,0] = pts[:,0]/5
    for i in range(pts.shape[0]):
        distance = cdist(pca1_f1, pts[i:i+1])
        idx = np.argmin(distance)
        plt.scatter(test_newData[idx, 0], f1_score[idx], s=144, marker = 'X',label = test_num_2_cityname[idx], color = color[i])
    print('pearson corre:', scipy.stats.pearsonr(test_newData[:, 0], f1_score))

    plt.yticks(fontsize=22)
    plt.xticks(fontsize=22)
    plt.xlabel('PCA1', font1)
    plt.ylabel('F1 score', font1)
    plt.legend(loc="best", fontsize=19, markerscale=0.98, ncol=1, labelspacing = 0.1, borderpad = 0.20)
    plt.tight_layout()
    plt.savefig('figures/pca1_vs_f1.png', bbox_inches='tight')
    plt.show()

def pca_visualize_center_radar(k):
    with open('results/training_set_index.txt') as json_file:
        city_index = json.load(json_file)
    data = np.zeros((len(city_index), 11))
    num_2_cityname = {}
    for city_num, city in enumerate(city_index):
        num_2_cityname[city_num] = city
        for idx_num, attribute in enumerate(city_index[city]):
            data[city_num, idx_num] = city_index[city][attribute]
    print('data shape: ', data.shape)

    data_mean = np.mean(data, axis=0, keepdims=True)
    data_std = np.std(data, axis=0, keepdims=True)
    data = (data - data_mean)/data_std

    k = k

    kmeanModel = KMeans(n_clusters=k, random_state = 1)
    kmeanModel.fit(data)

    # print(kmeanModel.labels_== 1)
    pca = PCA(n_components=6)
    newData = pca.fit_transform(data)

    change_order = True

    if change_order:
        SDNi = np.zeros(k)
        change_order_mapping = {}
        for i in range(k):
            SDNi[i] = np.mean(newData[kmeanModel.labels_== i][:,0])
        argsorted_SDNi = np.argsort(SDNi)
        for i in range(k):
            change_order_mapping[i] = np.where(argsorted_SDNi==i)[0][0]
        for i in range(len(kmeanModel.labels_)):
            kmeanModel.labels_[i] = change_order_mapping[kmeanModel.labels_[i]]

    cluster_centers_ = np.zeros((k, data.shape[1]))

    for i in range(k):
        # print(kmeanModel.labels_== i)
        cluster_centers_[i,:] = np.mean(data[kmeanModel.labels_== i], axis = 0, keepdims=True)
    # cluster_centers_ = cluster_centers_*data_std + data_mean
    name_list = ['avg degree', 'frc degree 1', 'frc degree 2', 'frc degree 3', 'frc degree 4', 'log circuity (r<0.5)', 'log circuity (r>0.5)', 'frc bridge edges', 'frc dead-end edges', 'frc bridge length', 'frc dead-end length']
    num_list = cluster_centers_[0]

    plt.style.use('ggplot')
    fig = plt.figure(figsize=(9.8, 7))#, dpi=1200)
    # polar coordinate
    ax = fig.add_subplot(111, polar=True)
    # data
    feature = name_list
    values = cluster_centers_[0]

    N = len(values)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    color = ['b', 'orange', 'g', 'r']
    for i in range(cluster_centers_.shape[0]):
        values = cluster_centers_[i]
        values = np.concatenate((values, [values[0]]))
        ax.plot(angles, values, 'o-', linewidth=2, label = 'Type '+ str(i+1), color = color[i])
        # fill color
        ax.fill(angles, values, alpha=0.1, color = color[i])


    ax.set_thetagrids(angles * 180 / np.pi, feature, fontsize = 16, color ='k')
    ax.grid(True)
    # show figure
    ax.yaxis.set_ticklabels([])
    plt.legend(loc ='best', bbox_to_anchor=(0.7, 0.65, 0.5, 0.5), fontsize = 18)
    plt.tight_layout()
    plt.savefig('figures/center.png', bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='get the measures for networks')
    parser.add_argument('--mode', default='', type=str)
    args = parser.parse_args()

    if not os.path.isdir('figures'):
        os.mkdir('figures')

    if args.mode == 'pca_visualize':
        pca_visualize(k=4)

    if args.mode == 'city_ratio':
        city_ratio(k=4)

    if args.mode == 'f1_vs_type':
        f1_vs_network_type(k=4)

    if args.mode == 'f1_vs_city':
        f1_vs_city()

    if args.mode == 'f1_vs_PCA1':
        f1_vs_PCA1()

    if args.mode == 'center':
        pca_visualize_center_radar(k=4)