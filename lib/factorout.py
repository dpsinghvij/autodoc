import re


class FactorDisp:
    def __init__(self,factors,value,evidence):
       self.factors= factors
       self.value= value
       self.evidence = evidence
       self.statement=""
       self.statement = "{}{} because of {}"
       split = self.factors[0].split('_')
       name = self.convert(split[0])
       if (split[1] == '0'):
           self.statement = self.statement.format("no ", self.convert(evidence), name)
       else:
           self.statement = self.statement.format("", self.convert(evidence), name)

    def add_evidence(self,evidence):
        self.evidence = evidence
        self.statement = "There is {} problem {} because of {}"
        split = self.factors[0].split('_')
        name = self.convert(split[0])
        if (split[1] == '0'):
            self.statement = self.statement.format("no",evidence, name)
        else:
            self.statement = self.statement.format("a", evidence,name)

    def convert(self,name):
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1).lower()

class FactorWithName:
    def __init__(self,probname,factordisp):
        self.prob_name= probname
        self.factordisp= factordisp
