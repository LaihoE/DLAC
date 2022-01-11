# Deep Learning Anti-Cheat For CSGO

## Work in progress  

Input the directory with your .dem files and the model outputs predictions for every shot during the game.

```python
from DLAC import Model

model = Model("./path_to_demos/")

model.predict_to_terminal(threshold=0.99)   # You can manually specify threshold, 0.95 by default

```
Other ways to output predictions  
model.predict_to_csv()  
model.predict_to_list()

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
# Architecture (totally not done in paint)
Code and training loop found in "training" directory.
## Current simple design
 
<img src="https://github.com/LaihoE/DLAC/blob/main/images/current.png?raw=true" width="400">  

Main problem with this one is that it does the predictions independent of each other so the model can't make predictions with full information. Will probably be superseded by below models.
## Multiple-kill-input GRU model

<img src="https://github.com/LaihoE/DLAC/blob/main/images/Gruception.png?raw=true" width="400">  

First iteration of this one seems to do similarly/better than the very optimized simple model.
## Transformer model

<img src="https://github.com/LaihoE/DLAC/blob/main/images/Transformer.png?raw=true" width="400">  

If we can feed it patches of words, images, sequences of speech pieces or (states, actions, rewards), why not sequences of kills?  
Currently not working too great. Could it be the fault of the implementer? 🤔. Very experimental, maybe replace GRU with simple linear layer or maybe CNN?

# Data
Around 500GB-1TB training data depending on how you slice the data, shape = (n_kills, timesteps, 6). The 6th variable is ID of player so that you can group together kills. 320 timesteps in total (currently only using 128).

## Speed
Parsing 100 MM demos using AMD Ryzen 9 5900x (12 cores 24 threads) and m2 SSD. 

In total: 41.57s  
Parsing: 20.70s    
Predicting: 20.87

This is done ONLY USING CPU, predictions can be sped up with GPU if needed.




## Special thank you to
Demoinfocs-golang is the underlying parser used for parsing the demos, found at:  
https://github.com/markus-wa/demoinfocs-golang.  

87andrewh has written the majority of the specific parser used, found at: https://github.com/87andrewh/DeepAimDetector/blob/master/parser/to_csv.go