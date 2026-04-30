# sdn-room-reverb
A first order sdn reverb implementation in python 

The code is implemented from the paper Efficient Synthesis of Room Acoustics via Scattering Delay Networks by Enzo De Sena et al. (2015), https://doi.org/10.1109/TASLP.2015.2438547 

This repository contains a Python implementation of a **Scattering Delay Network (SDN)** room reverberation model developed for the thesis:

> **Exploring Virtual Acoustics Algorithms in the SpatialScaper Framework: Effects on Sound Event Localization Model Training**

The implementation is independent of SpatialScaper and was created to support experiments involving SDN-based room impulse response (RIR) generation.


## Overview

This project implements an SDN-based approach to simulate room acoustics and generate Room Impulse Responses (RIRs). The model allows configurable room parameters and produces output suitable for acoustic analysis or convolution-based spatialization.

The repository includes:

- **`DST2FinalProject.ipynb`**  
  A Jupyter Notebook containing documentation, explanations, and example demonstrations of the SDN reverb class.

- **`DST2Final.py`**  
  A standalone Python script consolidating the notebook into a runnable implementation. Room parameters can be modified in the `main` section.


## Features

- SDN-based room impulse response generation  
- Configurable room and simulation parameters  
- RIR length and peak value reporting  
- Output `.wav` file generation  
- Example notebook demonstration  
- Lightweight implementation using standard scientific Python libraries  


## Dependencies

The implementation uses the following Python packages:

```python
numpy
soundfile
scipy
IPython
matplotlib
dataclasses
typing
```
Install the main dependencies with:
```python
pip install numpy soundfile scipy matplotlib ipython
```

## Usage
### Running the standalone script
From the repository directory, run:
```python
python DST2Final.py
```
The script will:
1. Generate an SDN-based room impulse response.
2. Print the RIR length.
3. Print the RIR peak value.
4. Save the generated impulse response as:
```python
sdn_rir.wav
```
in the current directory.

Notebook demonstration
To view the documented example workflow, open:
```python
DST2FinalProject.ipynb
```
Run the notebook cells sequentially to inspect the SDN reverb class, generate an example RIR, and listen to or visualize the output.

## Thesis Context
This implementation was developed as part of a thesis investigating how different virtual acoustic simulation methods affect Sound Event Localization and Detection (SELD) model performance.
The SDN approach was compared against:
Full-order Image Source Method (ISM)
Truncated ISM (max order)
Hybrid ISM with Ray Tracing (ISM+RT)

This repository provides a reproducible implementation of the SDN method used in the experimental evaluation.

## Reproducibility Notes
This code is intended as a research implementation, not a production-level acoustic simulation tool.

## Author
Xinran Chen
2026
