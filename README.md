# Deep Learning AntiCheat For CSGO

## Work in progress  

Input the directory with your .dem files and the model outputs predictions for every shot during the game.

```python
from DLAC import Model

model = Model("./path_to_demos/")
model.predict_to_terminal()     # Prints outputs

model.predict_to_terminal(threshold=0.99)   # You can manually specify threshold, 0.95 by default

# You can also get the outputs into a python list
outputs = model.predict_to_list()

# Or write them into a csv file
model.predict_to_csv("outputs.csv")
```

## Special thank you to
Demoinfocs-golang is the underlying parser used for parsing the demos, found at:  
https://github.com/markus-wa/demoinfocs-golang.  

87andrewh has written the majority of the specific parser used, found at: https://github.com/87andrewh/DeepAimDetector/blob/master/parser/to_csv.go