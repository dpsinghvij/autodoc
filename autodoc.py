from flask import Flask, render_template
from lib.data import Data
from lib.autobayes import BayesModel
import jsonpickle
from flask import request

app = Flask(__name__)
dataa = Data("data.csv")
bayesnet = BayesModel(dataa)

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/output.html')
def output():
    return render_template('output.html')

@app.route('/nodes')
def get_nodes():
    return jsonpickle.encode(dataa.getNodes(), unpicklable=False)


@app.route('/allpred', methods=['POST'])
def get_preds():
    requestdata = request.get_json()
    evidenceList = []
    for req in requestdata["evidence"]:
        evidenceList.append(req)
    heuristic= ""
    if "heuristic" in requestdata:
        heuristic= requestdata["heuristic"]
    return bayesnet.getAllProbabilities(evidenceList,heuristic)


@app.route('/pred', methods=['POST'])
def get_single_pred():
    requestdata = request.get_json()
    evidence_list = []
    heuristic = ""
    if "heuristic" in requestdata:
        heuristic = requestdata["heuristic"]
    for req in requestdata["evidence"]:
        evidence_list.append(req)
    query_list= []
    for q in requestdata["query"]:
        query_list.append(q)
    return bayesnet.getAskedProbability(evidence_list,query_list,heuristic)


if __name__ == '__main__':
    app.run(port=5001)
