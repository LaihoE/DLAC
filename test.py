from DLAC import Model

model = Model("./path_to_demos/")
model.predict_to_terminal()     # Prints outputs

model.predict_to_terminal(threshold=0.99)   # You can manually specify threshold, 0.95 by default

# You can also get the outputs into a python list
outputs = model.predict_to_list()

# Or write them into a csv file
model.predict_to_csv("outputs.csv")