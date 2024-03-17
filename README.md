# Rainfall Detection Model with Streamlit UI

This repository contains code for a machine learning model that predicts rainfall using historical weather data in Australia, along with a Streamlit user interface for easy interaction.

## Overview

Rainfall prediction is crucial for various sectors such as agriculture, disaster management, and urban planning. This project aims to provide an accurate rainfall prediction tool using machine learning techniques, specifically a Random Forest Classifier.

The model is trained on historical weather data from various locations in Australia, including features such as temperature, humidity, wind speed, and atmospheric pressure. It analyzes these features to predict whether it will rain tomorrow.

## Files Included

- **main.py**: Python script containing code for data preprocessing, model training, evaluation, and serialization.
- **app.py**: Streamlit application for user interaction and real-time prediction.
- **weatherAUS.csv**: Sample dataset containing historical weather data from Australia used for training and testing the model.
- **requirements.txt**: List of Python dependencies required to run the application.

## Usage

1. Clone the repository:

```bash
git clone https://github.com/your-username/rainfall-detection.git
```

2. Install dependencies:

```bash
cd rainfall-detection
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or submit a pull request.
