import re


class FactorDisp:
    def __init__(self,factors,value):
       self.factors= factors
       self.value= value
       self.statement=""
    
       split = factors[0].split('_')
       self.statement="Probability that there is a problem because of {} is {}"
       name= self.convert(split[0])
       if(split[1] == '0'):
            self.statement= self.statement.format(name,value)
       else:
           self.statement = self.statement.format(name, value)

    def convert(self,name):
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1).lower()

class FactorWithName:
    def __init__(self,probname,factordisp):
        self.prob_name= probname
        self.factordisp= factordisp