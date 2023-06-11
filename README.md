# ECG Age Regression

Implementation of ECG age regression.


## Data

CODE-15%: a large scale annotated dataset of 12-lead ECGs. 
https://zenodo.org/record/4916206

## Getting started
- Evalution
```
python evaluate.py ${CHECKPOINT}
```

- Train
```
python train.py
```

- Analysis

Survival analysis using difference of real age and predicted ECG age 

```
analysis.ipynb
```
