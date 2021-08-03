# Machine Learning AntiCheat For CSGO
You input the directory where your .dem files are, and the model will give you predictions of every shot that hits an enemy during the game.  
The code outputs a CSV file with the predictions.

## Quickstart Guide
You need to have Golang and clone Demoinfocs-golang (check how to setup from demoinfocs repo), found at: https://github.com/markus-wa/demoinfocs-golang

Currently only for demos that are recorded at 64-tick (NOT the same as the servers tickrate, for example normal MM demos are recorded at 32-tick).

Predict.py is currently the main file, found in the supervised folder. Just change the demo_folder to the folder where your .dem files are, and the code will output a CSV file in current directory with output like:  
Probability, Suspect Name, Suspect SteamId, Tick of the shot, .dem file name

## Current progress
Roughly 1/10 of the predictions are false positives at a 90% confidence threshold. Threshold is easy to change to your own liking. The model should be used more as a type of filter rather than a judge. **Should by no means be used to automatically ban people without human supervision!**  
Your best bet is to follow multiple games of the same player and take some type of average of them.

## Model
Catboost model (similar to XGBoost), deep learning doesn't seem to beat classical ML models at this task at < 500 000 datapoints (shots)

## Credits
Demoinfocs-golang is the underlying parser used for parsing the demos, found at: https://github.com/markus-wa/demoinfocs-golang.  
87andrewh has written the majority of the specific parser used, found at: https://github.com/87andrewh/DeepAimDetector/blob/master/parser/to_csv.go