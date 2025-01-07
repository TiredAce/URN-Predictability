import math
import random
import json
import copy
import timeit
import random
import numpy as np
import osmnx as ox
import networkx as nx
import requests
import geopandas as gpd
import matplotlib.cm as cm
from math import sin, cos, sqrt, atan2,radians
import matplotlib.colors as colors
ox.settings.log_console = True
import os
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-cn", "--cite_name",
        type = str,
        default = "New york",
    )

    parser.add_argument(
        "-lat",
        type = float,
        default = 40.73
    )

    parser.add_argument(
        "-lon",
        type = float,
        default= -73.94,
    )

    parser.add_argument(
        "--squareLength",
        type = int, 
        default = 20
    )

    parser.add_argument(
        "--trainSize",
        type = int, 
        default=300,
    )

    parser.add_argument(
        "--testSize",
        type = int,
        default = 100,
    )

    parser.add_argument(
        "--save_src",
        type = str,
        default = "./"
    )

    return parser.parse_args()

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:                  
        os.makedirs(path)            
        print('A new folder created.')
    else:
        print('Has already been created.')

def edges2dict(edgeFinal): 
    dicts = []
    n = len(edgeFinal)
    for i in range(0,n):
        dict1 = {'start':int(edgeFinal[i][0]), 'end':int(edgeFinal[i][1]),'inSample1':edgeFinal[i][3],'inSample2':edgeFinal[i][4]}
        dicts.append(dict1)
    return dicts

def nodes2dict(nodeFinal):
    dicts = []
    n = len(nodeFinal)
    for i in range(0,n):
        dict1 = {'osmid': int(nodeFinal[i][0]), 'lon': nodeFinal[i][1],'lat':nodeFinal[i][2]}
        dicts.append(dict1)
    return dicts

# function 1 for node merge
def distancePair(loc1,loc2):    #loc1:[lon1,lat1]; loc2:[lon2,lat2]
    R = 6373.0
    lon1, lat1 = radians(loc1[0]),radians(loc1[1])
    lon2, lat2 = radians(loc2[0]),radians(loc2[1])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (sin(dlat/2))**2 + cos(lat1) * cos(lat2) * (sin(dlon/2))**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    return distance

# function 2 for node merge
def neighborCell(nx,ny,x,y): # nx,ny: 100; x,y:0,1,2,...,100
    neighbor = list()
    xList = [x-1,x,x+1]
    yList = [y-1,y,y+1]
    for i in range(3):
        for j in range(3):
            if xList[i]>=0 and xList[i]<=nx and yList[j]>=0 and yList[j]<= ny:
                neighbor.append((xList[i],yList[j]))
    return neighbor

# the input is G1 (drive) and G6 (all private), and we need to generate a G = G1.union(G6) 
def OSMnx_graph(G1):  
    #======================================================= step 0
    #step_0.1 generate G: lo, la, nodeOsmid, edgeInfor
    G1 = ox.project_graph(G1, to_crs="EPSG:4326")
    G1nodes = list(G1.nodes(data = True))
    for i in range(len(G1nodes)):
        G1nodes[i][1]['osmid'] = G1nodes[i][0]
    G1_node = {str(G1nodes[i][1]['osmid']): (G1nodes[i][1]['x'],G1nodes[i][1]['y']) for i in range(len(G1nodes))} 
    G_node =copy.deepcopy(G1_node)
    """
    G_node 是一个字典 osmid: lon, lat
    """

    #step_0.2 edges    
    G1edges = list(G1.edges(data = True))
    G1_edge = {(int(G1edges[i][0]),int(G1edges[i][1])):1 for i in range(len(G1edges))} #1: drivable
    G_edge =copy.deepcopy(G1_edge)

    """
    G_edge 是一个字典 (osmid1, osmid2): 1
    """

    # step_0: get 1）node_osmid; 2) node logitude; 3) node latitude; 4) edge (from,to)
    #step_0.3 input
    lo = [G_node[i][0] for i in G_node.keys()]
    la = [G_node[i][1] for i in G_node.keys()]
    nodeOsmid = [int(i) for i in G_node.keys()]
    edgeInfor = [(i[0],i[1],G_edge[i]) for i in G_edge.keys()]        
    #  ===================================================================
    # step_1: decide nX, nY 
    loMin, loMax, laMin, laMax = np.min(lo), np.max(lo), np.min(la), np.max(la)
    R = 6373.0
    d = 0.03   # the merging threshold = 30 meters.  #change： July 15,2020
    dymax = 2*R*cos(laMax*math.pi/180.0)*math.pi/360.0
    nX = math.floor((loMax-loMin)*dymax/d) # 表示网格在纬度上的数量
    unitX = (loMax - loMin)/nX  # 单位网格的纬度上的长度
    nY = math.floor((laMax-laMin)*math.pi*R/(180.0*d)) # 表示网格在经度上的数量
    unitY = (laMax - laMin)/nY  # 单位网格的经度上的长度
    #  ===================================================================
    # step_2 go through all the nodes
    mapping = {}
    for i in range(nX+1):
        for j in range(nY+1):
            mapping[(i,j)]=list()
    for i in range(len(lo)):
        long, lati = lo[i], la[i]
        x = math.floor((long-loMin)*nX/(loMax-loMin))
        y = math.floor((lati-laMin)*nY/(laMax-laMin))
        mapping[(x,y)].append(i)
    #  ===================================================================
    # step_3 near relationship
    nearResult = list()
    for i in range(nX+1):
        for j in range(nY+1):
            count = (nY+1)*i+j
            neighbor = neighborCell(nX,nY,i,j)
            neighborNodes = list()
            for k in range(len(neighbor)):
                neighborNodes = neighborNodes + mapping[neighbor[k]]
            for n1 in range(len(mapping[(i,j)])):
                for n2 in range(len(neighborNodes)):
                    node1 = mapping[(i,j)][n1]
                    node2 = neighborNodes[n2]
                    loc1 = [lo[node1], la[node1]]
                    loc2 = [lo[node2], la[node2]]
                    if (distancePair(loc1,loc2)) < d:
                        if node1< node2 and (node1,node2) not in nearResult:
                            nearResult.append((node1,node2))
                        if node2< node1 and (node2,node1) not in nearResult:
                            nearResult.append((node2,node1))
    #  ===================================================================
    # step_4 merge operation
    nodeReach = {}
    for i in range(len(lo)):
        nodeReach[i]=[i]
    for k in range(len(nearResult)):
        i = nearResult[k][0]
        j = nearResult[k][1]
        iList = nodeReach[i]
        jList = nodeReach[j]
        ijList = list(set(iList).union(set(jList)))
        for p in range(len(ijList)):
            nodeReach[ijList[p]]=ijList
    # =====================================================================
    # step_5 get new information
    loNew = list()
    laNew = list()
    minOSMid = list()
    for i in range(len(lo)):
        xList = [lo[nodeReach[i][k]] for k in range(len(nodeReach[i]))]
        yList = [la[nodeReach[i][k]] for k in range(len(nodeReach[i]))]
        idList = [nodeOsmid[nodeReach[i][k]] for k in range(len(nodeReach[i]))]
        xAver = np.mean(xList)
        yAver = np.mean(yList)
        loNew.append(xAver)
        laNew.append(yAver)
        minOSMid.append(np.min(idList))
    # =====================================================================
    # step_6 get final node      
    nodeFinal = list()
    minOSMidClear = list(set(minOSMid))   # 合并后节点的数组
    for i in range(len(minOSMidClear)):
        indexGet = minOSMid.index(minOSMidClear[i])
        nodeFinal.append((minOSMid[indexGet],loNew[indexGet],laNew[indexGet]))            
    # =====================================================================
    # step_7 refresh the edge result    
    edgeListRaw = [(str(minOSMid[nodeOsmid.index(edgeInfor[i][0])]),str(minOSMid[nodeOsmid.index(edgeInfor[i][1])]),edgeInfor[i][2]) for i in range(len(edgeInfor))]
    edgeNearFinal = list(set(edgeListRaw))
    edgeFinal = list()
    for i in range(len(edgeNearFinal)):
        if int(edgeNearFinal[i][0])> int(edgeNearFinal[i][1]): 
            #we set the start point index of an edge is smaller than end point
            edgeFinal.append((edgeNearFinal[i][1],edgeNearFinal[i][0],edgeNearFinal[i][2]))
        else: 
            edgeFinal.append((edgeNearFinal[i][0],edgeNearFinal[i][1],edgeNearFinal[i][2]))
    edgeFinal = list(set(edgeFinal))
    # =====================================================================    
    # step_8 clear edgeFinal
    validIndex1 = list()
    nodeFinalList = [nodeFinal[i][0] for i in range(len(nodeFinal))]
    for i in range(len(edgeFinal)):
        if (int(edgeFinal[i][0]) in nodeFinalList) and (int(edgeFinal[i][1]) in nodeFinalList) :
            validIndex1.append(i)
    edgeFinalFinal = [edgeFinal[validIndex1[i]] for i in range(len(validIndex1))]
    # =====================================================================
    # step_9 clear nodeFinal
    nodeIdRaw = [nodeFinal[i][0] for i in range(len(nodeFinal))]
    nodeReceive = list()
    for i in range(len(edgeFinalFinal)):
        nodeReceive.append(int(edgeFinalFinal[i][0]))
        nodeReceive.append(int(edgeFinalFinal[i][1]))
    nodeReceive = list(set(nodeReceive))
    validIndex2 = list()
    for i in range(len(nodeIdRaw)):
        if (nodeIdRaw[i] in nodeReceive) :
            validIndex2.append(i)
    nodeFinalFinal = [nodeFinal[validIndex2[i]] for i in range(len(validIndex2))]
    return [nodeFinalFinal,edgeFinalFinal]


def sample(nodeInfor,edgeInfor):
    # step0: get the information
    nodeId = [nodeInfor[i][0] for i in range(len(nodeInfor))]
    longitude = [nodeInfor[i][1] for i in range(len(nodeInfor))]
    latitude = [nodeInfor[i][2] for i in range(len(nodeInfor))]
        
    # step1: generate the graph
    n = len(nodeId)
    A1 = np.array([[0] * n] * n)
    Graph1 = nx.Graph(A1)
    
    # step2: label
    column = [str(nodeId[i]) for i in range(n)]
    mapping = {0:str(nodeId[0])}
    for i in range(0,len(column)-1):
        mapping.setdefault(i+1,column[i+1])
    Graph1 = nx.relabel_nodes(Graph1,mapping)
    
    # step3: geolocation
    #POS = list()
    #for i in range(0,n):
    #    POS.append((float(longitude[i]),float(latitude[i])))
    #for i in range(0,n):
    #    Graph1.nodes[column[i]]['pos'] = POS[i]
    
    # step4: add edge
    edgeSet1 = list()
    for i in range(len(edgeInfor)):
        edgeRow = edgeInfor[i]
        edgeSet1.append((str(edgeRow[0]),str(edgeRow[1])))
    edgeSet = list(set(edgeSet1))
    Graph1.add_edges_from(edgeSet) 
    
    # step5: get the mininal spanning tree
    deleteNumber = int(len(Graph1.edges) * 0.20)
    
    T = nx.minimum_spanning_tree(Graph1)
    potentialDelete = list(set(Graph1.edges) - set(T.edges))
    #print ("potentialDelete",len(potentialDelete))
    realDelete1 = random.sample(potentialDelete, deleteNumber)
    realDelete2 = random.sample(potentialDelete, deleteNumber)
    print ("len(realDelete1)",len(realDelete1),"len(realDelete2)",len(realDelete2))
    
    # step6: prepare the output file
    edgeInforNew = list()
    for i in range(len(edgeInfor)):
        edgeRow = edgeInfor[i]
        item = list()
        if (str(edgeRow[0]),str(edgeRow[1])) in realDelete1 or (str(edgeRow[1]),str(edgeRow[0]))in realDelete1:
            item = [edgeRow[0],edgeRow[1],edgeRow[2],0]
        else:
            item = [edgeRow[0],edgeRow[1],edgeRow[2],1]
        if (str(edgeRow[0]),str(edgeRow[1])) in realDelete2 or (str(edgeRow[1]),str(edgeRow[0]))in realDelete2:
            item.append(0)
        else:
            item.append(1)
        edgeInforNew.append(item)
    
    #step7: transform the form
    returnEdgeInforNew = list()
    for i in range(len(edgeInforNew)):
        returnEdgeInforNew.append((edgeInforNew[i][0],edgeInforNew[i][1],edgeInforNew[i][2],edgeInforNew[i][3],edgeInforNew[i][4]))
    #print (returnEdgeInforNew)
    return returnEdgeInforNew


def getTrainIndex(n): #the input region is a n by n region.
    trainIndex = list()
    for i in range(n*n):
        row = math.floor(i/n)
        col = np.mod(i,n)
        if row%2 == 0 or col%2 == 0:
            trainIndex.append(i)
    return trainIndex

# 消除自环
def clearSameNodeEdge(edgeInfo):
    newEdgeInfo = list()
    for i in range(len(edgeInfo)):
        start = edgeInfo[i][0]
        end = edgeInfo[i][1]
        if start != end:    
            newEdgeInfo.append(edgeInfo[i])
    return newEdgeInfo



if __name__ == "__main__":
    
    args = get_args()

    squareLength = args.squareLength
    trainSize = args.trainSize
    testSize = args.testSize
    allIndex = list()

    fullList = list(range(squareLength * squareLength))
    train = getTrainIndex(squareLength)
    # random.shuffle(train)
    test = list(set(fullList) - set(train))
    # random.shuffle(test)
    allIndex = train + test

    pwd1 = args.save_src + "train/" + args.cite_name + "/"
    pwd2 = args.save_src = "test/" + args.cite_name + "/"

    mkdir(pwd1)
    mkdir(pwd2)

    distance = 500

    start0 = timeit.default_timer()
    countReal = 0
    count = 0

    lat = args.lat
    lon = args.lon
    location = (lat,lon)   
    LAT = squareLength*0.01     #longitude, latitude range
    LON = LAT 
    npd = squareLength
    dlat = 0.01    
    dlon = 0.01
    count_train, count_test = 0, 0
    ################ collect training data ###############################    
    for j1 in range(0, len(allIndex)):             #!!!!change trainSize,validateSize,test1Size,test2Size

        start1 = timeit.default_timer()
        j = allIndex[j1]                   #!!!!change 0,1,2,3
        row = math.floor(j/npd)
        col = np.mod(j,npd)
        lat1 = lat - 0.500*LAT + row*dlat*1.000
        lon1 = lon - 0.500*LON + col*dlon*1.000
        location1 = [lat1,lon1]
        print ("location1", location1)
        count += 1
        #distance = random.randint(min,max)
        try :
            G1 = ox.graph_from_point(location1, dist=distance, dist_type='bbox', network_type='drive')
            if len(G1.edges) == 0:  # 偶然会出现edge数量为0情况
                continue 
        except:
            print ("the graph is null")
        else:
            G1 = ox.project_graph(G1)
            if (len(G1)>10): 
                # merge the node
                graphResult = OSMnx_graph(G1)
                nodeFinal = graphResult[0]
                rawEdgeFinal = graphResult[1]
                print ("len(rawEdgeFinal)",len(rawEdgeFinal))
                rawEdgeFinal = clearSameNodeEdge(rawEdgeFinal)
                print ("len(rawEdgeFina)",len(rawEdgeFinal))
                #test whether it is ok to sample, edge num > 1.26 node num
                realEdgeFinal =  [(rawEdgeFinal[p][0],rawEdgeFinal[p][1]) for p in range(len(rawEdgeFinal))]
                realEdgeFinal = list(set(realEdgeFinal))
                if len(realEdgeFinal) > 1.26*len(nodeFinal):
                    if j1 < len(train):
                        filename = pwd1 + args.cite_name
                        count_train += 1
                    else:
                        filename = pwd2 + args.cite_name
                        count_test += 1
                    edgeFinal = sample(nodeFinal,rawEdgeFinal)
                    subfile = filename + str(j) +'nodes'+'.json'
                    nodefile = open(subfile,'w')
                    nodes = nodes2dict(nodeFinal)
                    json.dump(nodes,nodefile)
                    nodefile.close()

                    # save edges as a json file
                    subfile = filename + str(j) +'edges'+'.json'
                    edgefile = open(subfile,'w')
                    edges = edges2dict(edgeFinal)
                    json.dump(edges,edgefile)
                    edgefile.close()
                    countReal += 1
        print ("count",count,"      countReal",countReal)
        stop1 = timeit.default_timer()
        print('running time per iteration:', stop1 - start1)
        stop2 = timeit.default_timer()
        print('running time until now:', stop2 - start0)
        print ("========================================================")


    stop0 = timeit.default_timer()
    print('total running time:', stop0 - start0)
    
    print(f"{args.cite_name} - trainSize: {count_train} . testSize: {testSize}")

