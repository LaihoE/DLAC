# Mini version of VacNet
You feed the model ticks leading up to a single kill, and it gives you the probability of the kill being legit. 

## Got data?
If you happen to have demos (legit or cheating) pls share. CSGO saves demos to some extent in the folder:  
Counter-Strike Global Offensive\csgo\replays

## Current progress
Trained on 87k kills with accuracy 98 % on blatant cheaters (most of cheaters are spinbotting).

## Model
GRU that takes in data in the form of (1,256,4) 256 timesteps and 4 variables. (Changes often)