import joblib
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

model_path = "models_all_samples/model_733_samples.pkl"
log_folder = "data_filtered"

# Assuming you have already trained and saved the model
# Load the model from the saved file
loaded_model = joblib.load(model_path)

def read_log_file(file_path):
    # Read the data from the log file and return a numpy array of positions in meters
    # (assuming the data is a single-column containing the measured positions)
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("POS"):
                try:
                    d = line.strip().split(",")
                    x = [float(d[1]), float(d[2]), float(d[3])]
                    data.append(x)
                except:
                    print(line)
    return np.array(data)

def calculate_error(real_position, measured_positions):
    # Calculate the error in distance from the real position
    error_data = []
    for measured_position in measured_positions:
        error = np.linalg.norm(measured_position - real_position)
        error_data.append(error)
    return np.array(error_data)

def plot_gaussian_graph(errors, real_position, ax):
    # Fit Gaussian distribution to the error data
    mu, std = norm.fit(errors)
    x_range = np.linspace(min(errors), max(errors), 100)
    gaussian_curve = norm.pdf(x_range, mu, std)  # Probability Density Function

    ax.plot(x_range, gaussian_curve, label="Gaussian Distribution")
    ax.hist(errors, bins='auto', alpha=0.7, density=True, label="Error Histogram")

    ax.set_xlabel("Error in Distance from Real Position (m)")
    ax.set_ylabel("Occurrences (Density)")
    ax.set_title(f'Gaussian Graph for Real Position: {real_position}')
    ax.legend()
    ax.grid(True)

def main():
    log_files = [f for f in os.listdir(log_folder) if f.endswith(".log")]

    num_files = len(log_files)
    num_rows = 2  # You can adjust the number of rows and columns to control the layout
    num_cols = (num_files + num_rows - 1) // num_rows

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 10))
    axs = axs.ravel()  # Flatten the axes array to iterate easily


    for i, log_file in enumerate(log_files):
        # Extract x, y, and z from the log file name
        real_position = [float(val)/100 for val in os.path.splitext(os.path.basename(log_file))[0].split('_')]
     
        # Read data from the log file
        file_path = os.path.join(log_folder, log_file)
        measured_positions = read_log_file(file_path)

        predictions = loaded_model.predict(measured_positions)

        # Calculate errors in distance from the real position
        errors = calculate_error(real_position, predictions)
        print(errors)
        # Plot Gaussian graph for this log file
        plot_gaussian_graph(errors, real_position, axs[i])

    # Adjust the layout and show the graphs
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

