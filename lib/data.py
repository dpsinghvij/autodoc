import os
import csv
from lib.nodes import BayesNodes
import sys

#class for loading data
class Data:
    nodes = []
    childnodes = []

    #read csv file
    def __init__(self, filename):
        p = os.path.join(os.path.dirname(sys.argv[0]), filename)
        print(p)
        with open(p, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.nodes.append(BayesNodes(row[0], row[1], int(row[2])))

        print("data is loaded")

    #get the evidence
    def getNodes(self):
        return self.nodes

    #get the query
    def getChildNodes(self):
        childnodes= list(filter(lambda x: x.isSymptom == 0, self.nodes))
        return childnodes