# ECG Anomaly Detection with Temporal Convolutional Networks (TCN)

This project aims to detect anomalies in ECG data using a Temporal Convolutional Network (TCN) model. The project involves training a model, calculating anomaly scores, and visualizing detected anomalies in ECG data.

## Project Overview

The main steps in this project include:
1. **Data Preparation**: Splitting the ECG data into training, validation, and test sets, followed by normalization using a `RobustScaler`.
2. **Model Training**: Training a TCN model for anomaly detection and monitoring the training and validation losses.
3. **Anomaly Scoring**: Calculating anomaly scores on the test data and identifying anomalies using z-scores with a defined threshold.
4. **Chunk-Based Anomaly Detection**: Processing the data in chunks to handle larger sequences effectively.
5. **Visualization**: Plotting ECG data with highlighted normal and anomalous segments to visualize the detected anomalies.

## Installation

To run the project, you need to have the following Python packages installed:

- `numpy`
- `pandas`
- `matplotlib`
- `darts`

You can install the required packages using the following command:

```bash
pip install numpy pandas matplotlib darts
```

## Usage

1. **Model Training:**

   Train the TCN model using the provided training and validation data:

   ```python
   ecg_model.fit(train_series, val_series=val_series, trainer=trainer)
   ```

   Monitor the training and validation loss over epochs:

   ```python
   plt.figure(figsize=(10, 6))
   plt.plot(epochs, loss_callback.train_losses, label='Train Loss', marker='o')
   plt.plot(epochs, loss_callback.val_losses, label='Validation Loss', marker='o')
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.title('Training and Validation Loss Over Epochs')
   plt.legend()
   plt.grid(True)
   plt.show()
   ```

2. **Anomaly Scoring:**

   Calculate anomaly scores for the test data and plot the results:

   ```python
   anomaly_scores, model_forecasting = anomaly_model.score(test_series, start=START, return_model_prediction=True, verbose=True)
   plt.figure(figsize=(12, 6))
   plt.plot(time_index, scores, label="Anomaly Scores")
   plt.xlabel('Time')
   plt.ylabel('Score')
   plt.title('Anomaly Scores')
   plt.legend()
   plt.show()
   ```

   Detect anomalies using z-scores:

   ```python
   anomalies = z_scores > threshold
   ```

3. **Chunk-Based Anomaly Detection:**

   Process the data in chunks and visualize the detected anomalies:

   ```python
   chunk_anomalies, chunk_scores = calculate_anomaly_scores_by_chunk(time_index, scores, chunk_size)
   plt.figure(figsize=(12, 6))
   plt.plot(time_index, chunk_scores, label="Anomaly Scores")
   plt.scatter(time_index[chunk_anomalies], chunk_scores[chunk_anomalies], color='red', label="Detected Anomalies")
   plt.xlabel('Time')
   plt.ylabel('Score')
   plt.title('Anomaly Scores with Detected Anomalies (by Chunks)')
   plt.legend()
   plt.show()
   ```

4. **Visualization:**

   Plot the ECG data with normal and anomalous segments:

   ```python
   plot_ecg_with_anomalies(time_index, ecg_data, anomalies)
   ```

## Results

- The model successfully detects anomalies in the ECG data and provides a visual representation of the detected anomalies.
- The chunk-based approach enables effective processing of large sequences, identifying anomalies within smaller segments.

## Future Work

- Further fine-tuning of the model and scoring parameters.
- Integration of additional scoring methods and anomaly detection techniques.
- Exploration of different model architectures to improve anomaly detection performance.


## Acknowledgments

- The project utilizes the `Darts` library for time series modeling and the `matplotlib` library for visualization.
- Special thanks to the open-source community for providing valuable tools and resources.
