# Deep Learning Anti-Cheat For CSGO



Input the directory with your .dem files and the model outputs predictions for every shot during the game.

```python
from DLAC import Model

model = Model("./path_to_demos/")
model.predict_to_terminal(threshold=0.95)   # You can manually specify threshold, 0.95 by default
```
## Installation
Windows should be as easy as:
```python
pip install DLAC
```
Linux users will need to build the .so file. This requres GO.
```
git clone https://github.com/LaihoE/DLAC  
cd DLAC
python3 setup.py install
cd DLAC
go build -o parser.so -buildmode=c-shared
```

## You can choose between a bigger and a smaller model
```python
from DLAC import Model

model = Model("./path_to_demos/", model_type='big')
model.predict_to_terminal(threshold=0.99)   # 0.99 is recommended with the bigger model
```
The bigger model is slower with slightly better accuracy  


Other ways to output predictions  
model.predict_to_csv()  
model.predict_to_list()

## Example output from one shot  
```CSV
Name, Confidence of cheating, SteamId, File
PeskyCheater22, 0.9601634, 123456789, exampledemo.dem
```


## Special thank you to
Demoinfocs-golang is the underlying parser used for parsing the demos, found at:  
https://github.com/markus-wa/demoinfocs-golang.  

87andrewh has written the majority of the specific parser used, found at: https://github.com/87andrewh/DeepAimDetector/blob/master/parser/to_csv.go
