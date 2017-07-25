from pgmpy.models import BayesianModel


class BayesModel:
    """
    A class to handle bayesian model
    """

    def __init__(self, data):
        # bayesian model will be built here
        self.model = BayesianModel()
        self.data = data

    def get_name_from_id(self, id):
        for node in self.data.nodes:
            if node.id == id:
                return node.name
        return ""
