import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import joblib

model_path = "../models_all_samples/model_733_samples.pkl"
log_file = "mobile/ziczac_2.log"

# Assuming you have already trained and saved the model
# Load the model from the saved file
loaded_model = joblib.load(model_path)

# Read data from the log file
data = []
with open(log_file, 'r') as file:
    for line in file:
        try:
            _, x, y, z, _ = line.strip().split(',')
            data.append([float(x), float(y), float(z)])
        except ValueError:
            pass  # Skip lines with errors

predictions = loaded_model.predict(data)
x_pred, y_pred, z_pred = zip(*predictions)
# Unpack data into separate lists for x, y, and z coordinates
x_data, y_data, z_data = zip(*data)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the data
ax.scatter(x_data, y_data, z_data, c='b', marker='o')
ax.scatter(x_pred, y_pred, z_pred, c='r', marker='o')

# Set labels for the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()
