# Deep Learning AntiCheat For CSGO
You input the directory where your .dem files are, and the model will give you predictions of every shot that hits an enemy during the game.  
The code outputs a CSV file with the predictions.

```
from DLAC import Model

#model = Model("./path_to_demos/")
model.predict_to_terminal()     # Prints outputs
model.predict_to_terminal(threshold=0.99)   # You can manually specify threshold, 0.95 by default

# You can also get the outputs into a python list
outputs = model.predict_to_list()

# Or write them into a csv file
model.predict_to_csv("outputs.csv")
```

## Special thank you to
Demoinfocs-golang is the underlying parser used for parsing the demos, found at: https://github.com/markus-wa/demoinfocs-golang.  
87andrewh has written the majority of the specific parser used, found at: https://github.com/87andrewh/DeepAimDetector/blob/master/parser/to_csv.go