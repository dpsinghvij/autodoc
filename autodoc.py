from flask import Flask
from lib.data import Data
from lib.autobayes import BayesModel
import jsonpickle
from flask import request

app = Flask(__name__)
dataa = Data("data.csv")
bayesnet = BayesModel(dataa)


@app.route('/')
def hello_world():
    return 'Sunny Chutiya'


@app.route('/nodes')
def get_nodes():
    return jsonpickle.encode(dataa.getNodes(), unpicklable=False)


@app.route('/allpred', methods=['POST'])
def get_preds():
    requestdata = request.get_json()
    evidenceList = []
    for req in requestdata["evidence"]:
        evidenceList.append(req)

    return bayesnet.getAllProbabilities(evidenceList)


@app.route('/pred', methods=['POST'])
def get_single_pred():
    requestdata = request.get_json()
    evidence_list = []
    for req in requestdata["evidence"]:
        evidence_list.append(req)
    query_list= []
    for q in requestdata["query"]:
        query_list.append(q)
    return bayesnet.getAskedProbability(evidence_list,query_list)


if __name__ == '__main__':
    app.run()
