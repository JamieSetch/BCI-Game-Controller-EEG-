# EEG-Based Real-Time Brain-Computer Interface for Minecraft Control

This project implements a brain-computer interface (BCI) that processes EEG signals in real time to predict user commands and control Minecraft gameplay via keyboard input. It includes data loading, preprocessing, CNN model training, and live EEG streaming with command execution.
N.B. This is for an OpenBCI / MAC M1 Setup (not tested on windows or linux)
---

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Loading & Preprocessing](#data-loading--preprocessing)
  - [Model Training](#model-training)
  - [Real-Time EEG Streaming & Command Execution](#real-time-eeg-streaming--command-execution)
- [File Descriptions](#file-descriptions)
- [Requirements](#requirements)
- [Licence](#licence)

---

## Project Overview

This project utilises EEG data sampled at 250 Hz to train a convolutional neural network (CNN) model to classify brain states into actionable commands such as movement and actions in Minecraft. Real-time EEG data is streamed from an OpenBCI Cyton board using BrainFlow, filtered, features extracted, classified, and mapped to Minecraft controls via keyboard automation.

---

## Features

- Loading and windowing large EEG datasets from CSV files
- Data preprocessing including normalisation and label encoding
- CNN model architecture for EEG classification
- Real-time EEG data streaming via BrainFlow
- Bandpower feature extraction for classification
- Keyboard control mapping to Minecraft actions
- Modular, easily extendable design

---

## Installation

### 1. Clone the repository:
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo

### 2. Install Pyhton dependencies:
python -m venv venv         # Create virtual environment (optional but recommended)
source venv/bin/activate    # Activate virtual environment on Linux/Mac
# or
venv\Scripts\activate       # Activate virtual environment on Windows

pip install --upgrade pip
pip install -r requirements.txt

### 3. Set up hardware and drivers:
- Connect Cyton EEG board
- Ensure you have BrainFlow installed and working
- Adjust the serial port in your configuration if necessary
