# Deep Learning Anti-Cheat For CSGO

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

## Example output from one shot  
```CSV
Name, Confidence of cheating, SteamId, File
PeskyCheater22, 0.9601634, 123456789, exampledemo.dem
```
## Setup
Requires [Golang](https://golang.org/dl/)!
```
git clone https://github.com/LaihoE/DLAC  
cd DLAC
python setup.py install
```

## Performance
Parsing 100 MM demos using AMD Ryzen 9 5900x (12 cores 24 threads) and m2 SSD. 

In total: 41.57s  
Parsing: 20.70s    
Predicting: 20.87

This is done ONLY USING CPU, predictions can be sped up with GPU if needed.

## Special thank you to
Demoinfocs-golang is the underlying parser used for parsing the demos, found at:  
https://github.com/markus-wa/demoinfocs-golang.  

87andrewh has written the majority of the specific parser used, found at: https://github.com/87andrewh/DeepAimDetector/blob/master/parser/to_csv.go