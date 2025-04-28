# Neuroguessr

## Quickstart

### Requirements :

- nibabel==5.3.2
- pyinstaller==6.8.0


### Install

In the command line: 

```
cd
git clone https://github.com/FRramon/neuroguessr
```


#### For Linux : 

```
cd code
pyinstaller --name NeuroGuessr --windowed --add-data "../data:data"  --add-data "../code:code" -i "neuroguessr5.ico" neuroguessr.py
```

#### For MacOS : 

```
cd code
pyinstaller --name NeuroGuessr --windowed --add-data "../data:data"  --add-data "../code:code" -i "neuroguessr5.icns" neuroguessr.py
```
The app will be in dist/ directory, you can copy it to /Applications folder in the home directory.


If this does not work, you can still run the game from the command line with 

```
python neuroguessr.py
```



## Game summary

Neuroguessr is an education game for learing neuroanatomy


## Game modes

### Practice

In this game mode, the user can learn where are the regions (they blink after three try); and their structure/function briefly[^1].

### Contre la montre

In this game mode, the objective is to find all regions of the atlas in a minimal time

### Streak

In this game mode, the objective is to do the longest series of correct answers.


[^1] : Structure function summaries have been generated using the LLM Claude 3.7 Sonnet; there can be residual errors, hallucination.