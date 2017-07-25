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
    return 'Hello World!'


@app.route('/nodes')
def get_nodes():
    return jsonpickle.encode(dataa.getNodes(), unpicklable=False)


@app.route('/allpred', methods=['POST'])
def get_preds():
    requestdata = request.get_json()
    evidenceList = []
    for req in requestdata["evidence"]:
        evidenceList.append(req)
    print(evidenceList)
    return ""


@app.route('/pred', methods=['POST'])
def get_single_pred():
    requestdata = request.get_json()
    evidence_list = []
    for req in requestdata["evidence"]:
        evidence_list.append(req)
        print(bayesnet.get_name_from_id(req))
    query= requestdata["query"]
    print(evidence_list,query)
    return ""


if __name__ == '__main__':
    app.run()
