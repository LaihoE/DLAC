# Deep Learning AntiCheat For CSGO
You input the directory where your .dem files are, and the model will give you predictions of every shot that hits an enemy during the game.  
The code outputs a CSV file with the predictions.

## Special thank you to
Demoinfocs-golang is the underlying parser used for parsing the demos, found at: https://github.com/markus-wa/demoinfocs-golang.  
87andrewh has written the majority of the specific parser used, found at: https://github.com/87andrewh/DeepAimDetector/blob/master/parser/to_csv.go