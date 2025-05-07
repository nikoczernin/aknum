# AKNUM
## Reinforcement Learning

Die Lösungen zu den jeweiligen Tutorials sind in den gleichnamigen Python-Notebooks.  

## Aufbau des Repos
Die Environments sind vom Lernen unabhängige Klassen, allesamt Unterklassen der Klasse in `Environmetn.py`.  
Mit Environments interagiert immer ein Bot, wie aus `Bot.py`, 
dieser ist in der Lage eine oder mehrere Episoden auszuführen. 
Dies tut er mit seiner, bei der Erstellung initiieren Policy, die anfangs immer random ist.  
Die Policy eines Bots wird mit Lernalrogithmen verbessert, 
diese finden sich zB. in `MarkovDecisionProcess.py` oder `MonteCarlo.py`.  
