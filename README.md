# Mini version of VacNet
You feed the model 300 ticks leading up to a single kill, and it gives you the probability of the kill being legit. 

## Got data?
If you happen to have demos (legit or cheating) pls share. CSGO saves demos to some extent in the folder:  
Counter-Strike Global Offensive\csgo\replays

## Current progress
First model 80% validation accuracy trained on 1600 kills and tested on 100. (The kills used were very blatant)  
Model seems to struggle hard with legit cheaters :(

## Model
GRU that takes in data in the form of (1,300,20) 300 timesteps and 20 variables. (Changes often)