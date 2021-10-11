# Deep Learning AntiCheat For CSGO
You input the directory where your .dem files are, and the model will give you predictions of every shot that hits an enemy during the game.  
The code outputs a CSV file with the predictions.

## Quickstart Guide
You need to have Golang and clone Demoinfocs-golang (check how to setup from demoinfocs repo), found at: https://github.com/markus-wa/demoinfocs-golang  

Currently you should use the dem_to_pred.py file in the Deep_learning folder

## Model
GRU Model with 4 layers and hidden size of 128. Trained on 1.1 million datapoints (one datapoint = shot that has hit an enemy)
## Credits
Demoinfocs-golang is the underlying parser used for parsing the demos, found at: https://github.com/markus-wa/demoinfocs-golang.  
87andrewh has written the majority of the specific parser used, found at: https://github.com/87andrewh/DeepAimDetector/blob/master/parser/to_csv.go