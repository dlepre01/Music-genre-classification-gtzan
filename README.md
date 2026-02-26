# Music Genre Classification – GTZAN Dataset

##  Overview

This project implements a music genre classification pipeline using the GTZAN dataset (1,000 tracks, 10 genres).

The system combines audio feature engineering and supervised machine learning models, achieving up to 93% accuracy through ensemble methods.

Developed as part of a Deep Learning course project.  
This repository includes only the music classification task.


##  Objective

- Extract meaningful audio features from raw waveform signals
- Compare classical ML models and neural networks
- Evaluate ensemble learning strategies
- Build a demo music genre classifier



## Dataset

**GTZAN Dataset**
- 1,000 audio tracks
- 10 genres
- 30 seconds per track
- WAV format (22.05 kHz, mono)

Genres:
Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock


## Preprocessing Pipeline

- Audio segmentation: 12 overlapping 3-second segments per track
- Resampling and padding
- Standardization (StandardScaler)
- 80/20 train-test split


## Feature Engineering

Extracted features include:

- MFCC (Mel-Frequency Cepstral Coefficients)
- Spectral Centroid
- Spectral Bandwidth
- Spectral Rolloff
- Spectral Contrast
- Chroma Features (12 pitch classes)
- Zero Crossing Rate (ZCR)
- Root Mean Square Energy (RMS)
- Tempo (BPM estimation)

## Models Evaluated

| Model | Accuracy |
|-------|----------|
| SVM | 0.91 |
| Random Forest | 0.88 |
| KNN | 0.89 |
| XGBoost | 0.89 |
| Neural Network | 0.92 |
| Ensemble (soft voting) | **0.93** |

The ensemble model combining SVM, RF and KNN achieved the best overall performance.



## Tech Stack

- Python
- librosa
- scikit-learn
- XGBoost
- TensorFlow / Keras
- NumPy / Pandas


## Key Takeaway

Carefully engineered audio features combined with ensemble learning can achieve competitive performance without requiring deep end-to-end CNN architectures.
