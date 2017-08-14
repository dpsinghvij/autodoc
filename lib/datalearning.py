#import all the libraries
from random import *
import csv
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.estimators import BayesianEstimator
import pandas as pd
from collections import defaultdict
import numpy as np
import os, sys
import math

CSV_FILE = "alldata.csv"

#class for data learning and model implementation
class DataLearning:
    def __init__(self):

        #creating the bayesian model for data
        self.model_data = BayesianModel(
            [('cranksNormallyNotStarting', 'noFuelPressure'), ('cranksNormallyNotStarting', 'noSpark'),
             ('noSpark', 'sparkPlug'),
             ('cranksNormallyNotStarting', 'badTimingChain'), ('crankSlow', 'weakBattery'), ('crankSlow', 'badStarter'),
             ('crankSlow', 'corrodedBatteryTerminal'),
             ('vehicleBackFiring', 'badTimingChain'), ('vehicleBackFiring', 'badIgnitionSytem'),
             ('oneStrongClickOrKnock', 'badStarter'), ('oneStrongClickOrKnock', 'pistonNotWorking'),
             ('spinningWhinningOrGearGrinding', 'badStarter'), ('repeatingClickSound', 'weakBattery'),
             ('repeatingClickSound', 'badStarter'), ('repeatingClickSound', 'corrodedBatteryTerminal'),
             ('engineMisFiring', 'wornDistributor'), ('engineMisFiring', 'badIgnitionSytem'),
             ('engineVibration', 'harmonicBalancer'), ('engineVibration', 'wornEngineMounts'),
             ('vehicleRunsHot', 'faultyEngineCoolingFan'), ('vehicleRunsHot', 'brokenMissingFanAssembly'),
             ('overHeating', 'stuckThermostat'), ('overHeating', 'lowCoolantLevel'),
             ('overHeating', 'faultyEngineCoolingFan'),
             ('carbueratorStalling', 'badCarbuerator'), ('idleFluctuates', 'ignitionCoilForSpark'),
             ('idleFluctuates', 'cloggedAirFilter'), ('ignitionCoilForSpark', 'sparkPlug'),
             ('engineHesitation', 'faultyFuelFilter'), ('engineHesitation', 'cloggedAirFilter'),
             ('stalling', 'fuelPumpReplacement'), ('stalling', 'ignitionCoilForSpark'),
             ('roughRunningEngine', 'fuelSystemCleaning'), ('roughRunningEngine', 'ignitionCoilForSpark'),
             ('highIdle', 'faultyFuelFilter'), ('highIdle', 'vacuumLeaks'), ('ignitionMisFire', 'engineTuneUp'),
             ('ignitionMisFire', 'sparkPlug'),
             ('ignitionMisFire', 'ignitionCoilForSpark')])
        allNodes = self.model_data.nodes()
        self.create_mode()
        print(len(allNodes))


    def create_mode(self):

        #creating the bayesian model for expert
        self.model_expert = BayesianModel(
            [('cranksNormallyNotStarting', 'noFuelPressure'), ('cranksNormallyNotStarting', 'noSpark'),
             ('noSpark', 'sparkPlug'),
             ('cranksNormallyNotStarting', 'badTimingChain'), ('crankSlow', 'weakBattery'), ('crankSlow', 'badStarter'),
             ('crankSlow', 'corrodedBatteryTerminal'),
             ('vehicleBackFiring', 'badTimingChain'), ('vehicleBackFiring', 'badIgnitionSytem'),
             ('oneStrongClickOrKnock', 'badStarter'), ('oneStrongClickOrKnock', 'pistonNotWorking'),
             ('spinningWhinningOrGearGrinding', 'badStarter'), ('repeatingClickSound', 'weakBattery'),
             ('repeatingClickSound', 'badStarter'), ('repeatingClickSound', 'corrodedBatteryTerminal'),
             ('engineMisFiring', 'wornDistributor'), ('engineMisFiring', 'badIgnitionSytem'),
             ('engineVibration', 'harmonicBalancer'), ('engineVibration', 'wornEngineMounts'),
             ('vehicleRunsHot', 'faultyEngineCoolingFan'), ('vehicleRunsHot', 'brokenMissingFanAssembly'),
             ('overHeating', 'stuckThermostat'), ('overHeating', 'lowCoolantLevel'),
             ('overHeating', 'faultyEngineCoolingFan'),
             ('carbueratorStalling', 'badCarbuerator'), ('idleFluctuates', 'ignitionCoilForSpark'),
             ('idleFluctuates', 'cloggedAirFilter'), ('ignitionCoilForSpark', 'sparkPlug'),
             ('engineHesitation', 'faultyFuelFilter'), ('engineHesitation', 'cloggedAirFilter'),
             ('stalling', 'fuelPumpReplacement'), ('stalling', 'ignitionCoilForSpark'),
             ('roughRunningEngine', 'fuelSystemCleaning'), ('roughRunningEngine', 'ignitionCoilForSpark'),
             ('highIdle', 'faultyFuelFilter'), ('highIdle', 'vacuumLeaks'), ('ignitionMisFire', 'engineTuneUp'),
             ('ignitionMisFire', 'sparkPlug'),
             ('ignitionMisFire', 'ignitionCoilForSpark')])


        #CPD for root nodes
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

        #assign CPDs into the model
        self.model_expert.add_cpds(cpd_carbueratorStalling, cpd_crankSlow, cpd_cranksNormallyNotStarting,
                              cpd_engineHesitation, cpd_engineMisFiring, cpd_engineVibration, cpd_highIdle,
                              cpd_idleFluctuates,
                              cpd_ignitionMisFire, cpd_oneStrongClickOrKnock, cpd_overHeating, cpd_repeatingClickSound,
                              cpd_roughRunningEngine, cpd_spinningWhinningOrGearGrinding, cpd_stalling,
                              cpd_vehicleBackFiring,
                              cpd_vehicleRunsHot, cpd_badCarbuerator, cpd_weakBattery, cpd_badStarter,
                              cpd_noFuelPressure,
                              cpd_faultyFuelFilter, cpd_cloggedAirFilter, cpd_wornDistributor, cpd_wornEngineMounts,
                              cpd_harmonicBalancer, cpd_vacuumLeaks, cpd_engineTuneUp, cpd_sparkPlug,
                              cpd_pistonNotWorking,
                              cpd_lowCoolantLevel, cpd_faultyEngineCoolingFan, cpd_stuckThermostat,
                              cpd_corrodedBatteryTerminal,
                              cpd_fuelSystemCleaning, cpd_fuelPumpReplacement, cpd_badIgnitionSytem, cpd_badTimingChain,
                              cpd_brokenMissingFanAssembly, cpd_noSpark, cpd_ignitionCoilForSpark)

        self.model_expert.check_model()

    #generate data file when the model is created
    def generate_data_file(self):
        path = self.getPathWithFileName(CSV_FILE)
        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow((self.model_data.nodes()))

    #write random binary data into the CSV file
    def add_data_to_file(self):
        path = self.getPathWithFileName(CSV_FILE)
        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow((self.model_data.nodes()))
        randBinList = lambda n: [randint(0, 1) for b in range(1, n + 1)]
        for i in range(200000):
            data = randBinList(41)
            with open(path, 'a') as f:
                a = csv.writer(f, quoting=csv.QUOTE_ALL)
                a.writerow(data)

    #bayesian estimator for learning CPDs from the user data
    def learn_from_data(self):
        data= pd.read_csv(self.getPathWithFileName(CSV_FILE))
        self.model_data.fit(data, estimator=BayesianEstimator, prior_type="BDeu")

    #get path
    def getPathWithFileName(self, filename=""):
        return os.path.join(os.path.dirname(sys.argv[0]), filename)

    #count the number of rows in the CSV file and group by nodes
    def get_lookup(self,node):
        meAndMyParents = list(self.model_data._get_ancestors_of(node))
        df = pd.read_csv(self.getPathWithFileName(CSV_FILE))
        if (len(meAndMyParents) == 1):

            df = df.groupby(meAndMyParents).agg({meAndMyParents[0]: 'count'})
            dicValue = {'0': df.iloc[:, -1][0], '1': df.iloc[:, -1][1]}

        else:
            meAndMyParents.remove(node)
            df = df.groupby(meAndMyParents).agg({node: 'count'}).reset_index().rename(
                columns={node: 'countsym'})

            dicValue = {}
            for i in df.index:

                rightMostCol = df.ix[i]['countsym']
                my_lst = list(df.ix[i])
                my_lst = my_lst[:-1]
                my_lst_str = ''.join(map(str, my_lst))
                print(my_lst_str)
                dicValue.update({my_lst_str: rightMostCol})
                print(dicValue)
        return dicValue

    #learning function calculation
    def change_expert_for_data(self):
        for node in self.model_data.cpds:
            dic_value= self.get_lookup(node.variable)
            it = np.nditer(node.values, flags=['multi_index'])
            while not it.finished:
                print("%f <%s>" % (it.value, it.multi_index))
                index = []
                for y in it.multi_index:
                    temp = []
                    temp.append(y)
                    index.append(temp)
                meAndMyParents = list(self.model_data._get_ancestors_of(node.variable))
                if(len(meAndMyParents) == 1):
                    lookup=''.join(str(x) for x in np.asarray(it.multi_index))
                else:
                    lookup = ''.join(str(x) for x in np.asarray(it.multi_index)[1:])

                num = dic_value.get(lookup)
                if(num is None):
                    num=0
                x = 1 / (1 + math.exp(-((num *.01) - 6)))
                weightdata = x / (1 + x)
                weightexpert = 1 - weightdata
                self.model_expert.get_cpds(node.variable).values[index]= \
                    self.model_expert.get_cpds(node.variable).values[index]*weightexpert +\
                    self.model_data.get_cpds(node.variable).values[index]*weightdata

                print(weightdata,weightexpert)
                it.iternext()

        print(self.model_expert.get_cpds())