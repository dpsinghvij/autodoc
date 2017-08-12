
#importing the library
import json

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from itertools import product
from pgmpy.inference.EliminationOrder import MinFill, WeightedMinFill, MinWeight, MinNeighbours
from pgmpy.extern import six

from lib.factorout import FactorDisp, FactorWithName


class BayesModel:
    """
    A class to handle bayesian model
    """

    def __init__(self, data):
        # bayesian model will be built here
        self.model = self.createModel()
        self.loadCpds()
        #check for errors
        # validate model
        assert self.model.check_model()
        # applying inference
        self.infer = VariableElimination(self.model)
        self.data = data

    def getAllProbabilities(self,evidences,algorithm= 'min_fill'):

        user_evidence= []
        for evidence in evidences:
            user_evidence.append(self.get_name_from_id(evidence))
        #print(user_evidence)

        for i in range(len(user_evidence)):
            activeTrailNodes = self.model.active_trail_nodes(user_evidence[i])
            #print(activeTrailNodes)
            for key, value in activeTrailNodes.items():
                nodes = list(value)
            #print("printing..", nodes)
            output = []
            for j in range(len(nodes)):
                #print(nodes[j])

                if nodes[j] != user_evidence[i]:
                    ordering = self.get_elimination_order([nodes[j]],{user_evidence[i]: 1},algo= algorithm)
                    print(ordering)
                    q = self.infer.query(variables=[nodes[j]], evidence={user_evidence[i]: 1}, elimination_order=ordering)
                    output.append(FactorWithName(nodes[j], self.convertFactorToDict(q[nodes[j]])))
        return json.dumps(output,default=self.obj_dict)

    # function for getting the CPDs for selected nodes with
    def getAskedProbability(self,evidences, queries,algorithm= 'min_fill'):
        user_evidence=[]
        output = []
        for evidence in evidences:
            user_evidence.append(self.get_name_from_id(evidence))

        query_variable=[]
        for query in queries:
            query_variable.append(self.get_name_from_id(query))

        for i in range(len(user_evidence)):
            for j in range(len(query_variable)):
                ordering = self.get_elimination_order([query_variable[j]], {user_evidence[i]: 1}, algo=algorithm)
                print(ordering)
                q = self.infer.query(variables=[query_variable[j]], evidence={user_evidence[i]: 1}, elimination_order=ordering)
                output.append(FactorWithName(query_variable[j], self.convertFactorToDict(q[query_variable[j]])))

        return json.dumps(output, default=self.obj_dict)


    def get_elimination_order(self,variables,evidence,algo='min_fill'):
        elimination_order = list(set(self.infer.variables) -
                                 set(variables) -
                                 set(evidence.keys() if evidence else []))
        if(algo=='min_fill'):
            elimination_order= MinFill(self.model).get_elimination_order(elimination_order)
        elif(algo=='min_weight'):
            elimination_order = MinWeight(self.model).get_elimination_order(elimination_order)
        elif(algo=='min_neighbor'):
            elimination_order = MinNeighbours(self.model).get_elimination_order(elimination_order)
        elif(algo=='weighted_min_fill'):
            elimination_order = WeightedMinFill(self.model).get_elimination_order(elimination_order)

        return elimination_order


    def createModel(self):
        return BayesianModel([('cranksNormallyNotStarting', 'noFuelPressure'), ('cranksNormallyNotStarting', 'noSpark'),
                              ('noSpark', 'sparkPlug'),
                              ('cranksNormallyNotStarting', 'badTimingChain'), ('crankSlow', 'weakBattery'),
                              ('crankSlow', 'badStarter'),
                              ('crankSlow', 'corrodedBatteryTerminal'),
                              ('vehicleBackFiring', 'badTimingChain'), ('vehicleBackFiring', 'badIgnitionSytem'),
                              ('oneStrongClickOrKnock', 'badStarter'), ('oneStrongClickOrKnock', 'pistonNotWorking'),
                              ('spinningWhinningOrGearGrinding', 'badStarter'), ('repeatingClickSound', 'weakBattery'),
                              ('repeatingClickSound', 'badStarter'), ('repeatingClickSound', 'corrodedBatteryTerminal'),
                              ('engineMisFiring', 'wornDistributor'), ('engineMisFiring', 'badIgnitionSytem'),
                              ('engineVibration', 'harmonicBalancer'), ('engineVibration', 'wornEngineMounts'),
                              ('vehicleRunsHot', 'faultyEngineCoolingFan'),
                              ('vehicleRunsHot', 'brokenMissingFanAssembly'),
                              ('overHeating', 'stuckThermostat'), ('overHeating', 'lowCoolantLevel'),
                              ('overHeating', 'faultyEngineCoolingFan'),
                              ('carbueratorStalling', 'badCarbuerator'), ('idleFluctuates', 'ignitionCoilForSpark'),
                              ('idleFluctuates', 'cloggedAirFilter'), ('ignitionCoilForSpark', 'sparkPlug'),
                              ('engineHesitation', 'faultyFuelFilter'), ('engineHesitation', 'cloggedAirFilter'),
                              ('stalling', 'fuelPumpReplacement'), ('stalling', 'ignitionCoilForSpark'),
                              ('roughRunningEngine', 'fuelSystemCleaning'),
                              ('roughRunningEngine', 'ignitionCoilForSpark'),
                              ('highIdle', 'faultyFuelFilter'), ('highIdle', 'vacuumLeaks'),
                              ('ignitionMisFire', 'engineTuneUp'), ('ignitionMisFire', 'sparkPlug'),
                              ('ignitionMisFire', 'ignitionCoilForSpark')])

    def get_name_from_id(self, id):
        for node in self.data.nodes:
            if node.id == id:
                return node.name
        return ""





    def loadCpds(self):
        # cpds for roots

        cpd_carbueratorStalling = TabularCPD(variable='carbueratorStalling', variable_card=2, values=[[0.3, 0.7]])

        cpd_crankSlow = TabularCPD(variable='crankSlow', variable_card=2, values=[[0.4, 0.6]])

        cpd_cranksNormallyNotStarting = TabularCPD(variable='cranksNormallyNotStarting', variable_card=2,
                                                   values=[[0.5, 0.5]])

        cpd_engineHesitation = TabularCPD(variable='engineHesitation', variable_card=2, values=[[0.65, 0.35]])

        cpd_engineMisFiring = TabularCPD(variable='engineMisFiring', variable_card=2, values=[[0.35, 0.65]])

        cpd_engineVibration = TabularCPD(variable='engineVibration', variable_card=2, values=[[0.55, 0.45]])

        cpd_highIdle = TabularCPD(variable='highIdle', variable_card=2, values=[[0.48, 0.52]])

        cpd_idleFluctuates = TabularCPD(variable='idleFluctuates', variable_card=2, values=[[0.4, 0.6]])

        cpd_ignitionMisFire = TabularCPD(variable='ignitionMisFire', variable_card=2, values=[[0.35, 0.65]])

        cpd_oneStrongClickOrKnock = TabularCPD(variable='oneStrongClickOrKnock', variable_card=2, values=[[0.8, 0.2]])

        cpd_overHeating = TabularCPD(variable='overHeating', variable_card=2, values=[[0.3, 0.7]])

        cpd_repeatingClickSound = TabularCPD(variable='repeatingClickSound', variable_card=2, values=[[0.4, 0.6]])

        cpd_roughRunningEngine = TabularCPD(variable='roughRunningEngine', variable_card=2, values=[[0.5, 0.5]])

        cpd_spinningWhinningOrGearGrinding = TabularCPD(variable='spinningWhinningOrGearGrinding', variable_card=2,
                                                        values=[[0.75, 0.25]])

        cpd_stalling = TabularCPD(variable='stalling', variable_card=2, values=[[0.5, 0.5]])

        cpd_vehicleBackFiring = TabularCPD(variable='vehicleBackFiring', variable_card=2, values=[[0.6, 0.4]])

        cpd_vehicleRunsHot = TabularCPD(variable='vehicleRunsHot', variable_card=2, values=[[0.3, 0.7]])

        # cpds for leaves

        cpd_badCarbuerator = TabularCPD(variable='badCarbuerator', variable_card=2,
                                        values=[[0.65, 0.10],
                                                [0.35, 0.90]],
                                        evidence=['carbueratorStalling'],
                                        evidence_card=[2])

        cpd_weakBattery = TabularCPD(variable='weakBattery', variable_card=2,
                                     values=[[0.9, 0.3, 0.4, 0.2],
                                             [0.1, 0.7, 0.6, 0.8]],
                                     evidence=['crankSlow', 'repeatingClickSound'],
                                     evidence_card=[2, 2])

        cpd_badStarter = TabularCPD(variable='badStarter', variable_card=2,
                                    values=[
                                        [0.9, 0.8, 0.85, 0.55, 0.7, 0.6, 0.5, 0.2, 0.3, 0.45, 0.6, 0.1, 0.4, 0.25, 0.15,
                                         0.05],
                                        [0.1, 0.2, 0.15, 0.45, 0.3, 0.4, 0.5, 0.8, 0.7, 0.55, 0.4, 0.9, 0.6, 0.75, 0.85,
                                         0.95]],
                                    evidence=['crankSlow', 'oneStrongClickOrKnock', 'spinningWhinningOrGearGrinding',
                                              'repeatingClickSound'],
                                    evidence_card=[2, 2, 2, 2])

        cpd_noFuelPressure = TabularCPD(variable='noFuelPressure', variable_card=2,
                                        values=[[0.65, 0.10],
                                                [0.35, 0.90]],
                                        evidence=['cranksNormallyNotStarting'],
                                        evidence_card=[2])

        cpd_faultyFuelFilter = TabularCPD(variable='faultyFuelFilter', variable_card=2,
                                          values=[[0.9, 0.3, 0.4, 0.2],
                                                  [0.1, 0.7, 0.6, 0.8]],
                                          evidence=['engineHesitation', 'highIdle'],
                                          evidence_card=[2, 2])

        cpd_cloggedAirFilter = TabularCPD(variable='cloggedAirFilter', variable_card=2,
                                          values=[[0.9, 0.3, 0.4, 0.2],
                                                  [0.1, 0.7, 0.6, 0.8]],
                                          evidence=['idleFluctuates', 'engineHesitation'],
                                          evidence_card=[2, 2])

        cpd_wornDistributor = TabularCPD(variable='wornDistributor', variable_card=2,
                                         values=[[0.65, 0.10],
                                                 [0.35, 0.90]],
                                         evidence=['engineMisFiring'],
                                         evidence_card=[2])

        cpd_wornEngineMounts = TabularCPD(variable='wornEngineMounts', variable_card=2,
                                          values=[[0.65, 0.10],
                                                  [0.35, 0.90]],
                                          evidence=['engineVibration'],
                                          evidence_card=[2])

        cpd_harmonicBalancer = TabularCPD(variable='harmonicBalancer', variable_card=2,
                                          values=[[0.65, 0.10],
                                                  [0.35, 0.90]],
                                          evidence=['engineVibration'],
                                          evidence_card=[2])

        cpd_vacuumLeaks = TabularCPD(variable='vacuumLeaks', variable_card=2,
                                     values=[[0.65, 0.10],
                                             [0.35, 0.90]],
                                     evidence=['highIdle'],
                                     evidence_card=[2])

        cpd_engineTuneUp = TabularCPD(variable='engineTuneUp', variable_card=2,
                                      values=[[0.65, 0.10],
                                              [0.35, 0.90]],
                                      evidence=['ignitionMisFire'],
                                      evidence_card=[2])

        cpd_sparkPlug = TabularCPD(variable='sparkPlug', variable_card=2,
                                   values=[[0.90, 0.80, 0.85, 0.40, 0.20, 0.25, 0.15, 0.05],
                                           [0.10, 0.20, 0.15, 0.60, 0.80, 0.75, 0.85, 0.95]],
                                   evidence=['noSpark', 'ignitionCoilForSpark', 'ignitionMisFire'],
                                   evidence_card=[2, 2, 2])

        cpd_pistonNotWorking = TabularCPD(variable='pistonNotWorking', variable_card=2,
                                          values=[[0.65, 0.10],
                                                  [0.35, 0.90]],
                                          evidence=['oneStrongClickOrKnock'],
                                          evidence_card=[2])

        cpd_lowCoolantLevel = TabularCPD(variable='lowCoolantLevel', variable_card=2,
                                         values=[[0.65, 0.10],
                                                 [0.35, 0.90]],
                                         evidence=['overHeating'],
                                         evidence_card=[2])

        cpd_faultyEngineCoolingFan = TabularCPD(variable='faultyEngineCoolingFan', variable_card=2,
                                                values=[[0.9, 0.3, 0.4, 0.2],
                                                        [0.1, 0.7, 0.6, 0.8]],
                                                evidence=['vehicleRunsHot', 'overHeating'],
                                                evidence_card=[2, 2])

        cpd_stuckThermostat = TabularCPD(variable='stuckThermostat', variable_card=2,
                                         values=[[0.65, 0.10],
                                                 [0.35, 0.90]],
                                         evidence=['overHeating'],
                                         evidence_card=[2])

        cpd_corrodedBatteryTerminal = TabularCPD(variable='corrodedBatteryTerminal', variable_card=2,
                                                 values=[[0.9, 0.3, 0.4, 0.2],
                                                         [0.1, 0.7, 0.6, 0.8]],
                                                 evidence=['crankSlow', 'repeatingClickSound'],
                                                 evidence_card=[2, 2])

        cpd_fuelSystemCleaning = TabularCPD(variable='fuelSystemCleaning', variable_card=2,
                                            values=[[0.65, 0.10],
                                                    [0.35, 0.90]],
                                            evidence=['roughRunningEngine'],
                                            evidence_card=[2])

        cpd_fuelPumpReplacement = TabularCPD(variable='fuelPumpReplacement', variable_card=2,
                                             values=[[0.65, 0.10],
                                                     [0.35, 0.90]],
                                             evidence=['stalling'],
                                             evidence_card=[2])

        cpd_badIgnitionSytem = TabularCPD(variable='badIgnitionSytem', variable_card=2,
                                          values=[[0.9, 0.3, 0.4, 0.2],
                                                  [0.1, 0.7, 0.6, 0.8]],
                                          evidence=['vehicleBackFiring', 'engineMisFiring'],
                                          evidence_card=[2, 2])

        cpd_badTimingChain = TabularCPD(variable='badTimingChain', variable_card=2,
                                        values=[[0.9, 0.3, 0.4, 0.2],
                                                [0.1, 0.7, 0.6, 0.8]],
                                        evidence=['cranksNormallyNotStarting', 'vehicleBackFiring'],
                                        evidence_card=[2, 2])

        cpd_brokenMissingFanAssembly = TabularCPD(variable='brokenMissingFanAssembly', variable_card=2,
                                                  values=[[0.65, 0.10],
                                                          [0.35, 0.90]],
                                                  evidence=['vehicleRunsHot'],
                                                  evidence_card=[2])

        # cpds for intermediate nodes


        cpd_noSpark = TabularCPD(variable='noSpark', variable_card=2,
                                 values=[[0.65, 0.10],
                                         [0.35, 0.90]],
                                 evidence=['cranksNormallyNotStarting'],
                                 evidence_card=[2])

        cpd_ignitionCoilForSpark = TabularCPD(variable='ignitionCoilForSpark', variable_card=2,
                                              values=[
                                                  [0.9, 0.8, 0.85, 0.55, 0.7, 0.6, 0.5, 0.2, 0.3, 0.45, 0.6, 0.1, 0.4,
                                                   0.25, 0.15, 0.05],
                                                  [0.1, 0.2, 0.15, 0.45, 0.3, 0.4, 0.5, 0.8, 0.7, 0.55, 0.4, 0.9, 0.6,
                                                   0.75, 0.85, 0.95]],
                                              evidence=['idleFluctuates', 'stalling', 'roughRunningEngine',
                                                        'ignitionMisFire'],
                                              evidence_card=[2, 2, 2, 2])

        # adding CPDs to the model
        self.model.add_cpds(cpd_carbueratorStalling, cpd_crankSlow, cpd_cranksNormallyNotStarting,
                       cpd_engineHesitation, cpd_engineMisFiring, cpd_engineVibration, cpd_highIdle, cpd_idleFluctuates,
                       cpd_ignitionMisFire, cpd_oneStrongClickOrKnock, cpd_overHeating, cpd_repeatingClickSound,
                       cpd_roughRunningEngine, cpd_spinningWhinningOrGearGrinding, cpd_stalling, cpd_vehicleBackFiring,
                       cpd_vehicleRunsHot, cpd_badCarbuerator, cpd_weakBattery, cpd_badStarter, cpd_noFuelPressure,
                       cpd_faultyFuelFilter, cpd_cloggedAirFilter, cpd_wornDistributor, cpd_wornEngineMounts,
                       cpd_harmonicBalancer, cpd_vacuumLeaks, cpd_engineTuneUp, cpd_sparkPlug, cpd_pistonNotWorking,
                       cpd_lowCoolantLevel, cpd_faultyEngineCoolingFan, cpd_stuckThermostat,
                       cpd_corrodedBatteryTerminal,
                       cpd_fuelSystemCleaning, cpd_fuelPumpReplacement, cpd_badIgnitionSytem, cpd_badTimingChain,
                       cpd_brokenMissingFanAssembly, cpd_noSpark, cpd_ignitionCoilForSpark)

    def obj_dict(self,obj):
        return obj.__dict__

    def convertFactorToDict(self,factor):
        dic = {}
        string_header = list(map(lambda x: six.text_type(x), factor.scope()))
        string_header.append('{phi_or_p}({variables})'.format(phi_or_p="phi",
                                                              variables=','.join(string_header)))

        value_index = 0
        factor_table = []
        factordisp = []
        for prob in product(*[range(card) for card in factor.cardinality]):
            prob_list = ["{s}_{d}".format(s=list(factor.variables)[i], d=prob[i])
                         for i in range(len(factor.variables))]
            # prob_list.append(factor.values.ravel()[value_index])

            # factor_table.append(prob_list)
            factordisp.append(FactorDisp(prob_list, factor.values.ravel()[value_index]))
            value_index += 1
        return factordisp