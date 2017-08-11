from lib.datalearning import DataLearning

data= DataLearning()
#data.generate_data_file()
data.add_data_to_file()
data.learn_from_data()
#data.get_lookup('lowCoolantLevel')
data.change_expert_for_data()