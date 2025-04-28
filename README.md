# Neuroguessr

## Quickstart

### Requirements :

- nibabel==5.3.2
- pyinstaller==6.8.0
- PyQt5==5.15.11

(neuroguessr cannot work with pyqt6)

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

Neuroguessr is an educational game designed to help learning neuroanatomy.

## Game Modes

### Practice

In this mode, users can learn brain regions at their own pace. Regions will blink after three attempts to assist learning. Each region includes brief information about its structure and function.*

### Contre la montre :racing_car:

Race against the clock to identify all regions in the atlas as quickly as possible. 

### Streak :fire:

Test your knowledge by aiming for the longest consecutive series of correct answers. 

## Display Options

Users can choose to view the atlas with color-coded regions or in grayscale, allowing for different learning approaches and difficulty levels.

## Available Atlases

Neuroguessr offers a comprehensive range of anatomical atlases to explore:

### Cortical Atlases

- AAL (Automated Anatomical Labeling)
- Brodmann Areas
- Harvard-Oxford Cortical Atlas

### Subcortical Structures


### Specialized subfield atlases:

- Thalamus 
- Hippocampus
- Amygdala

### White Matter Tracts

### Cerebellum


*Structure and function summaries were generated using the LLM Claude 3.7 Sonnet. There might be errors.