
import numpy as np

# -------------------------------
# Step 1: Input Data (Iteration 1)
# -------------------------------
# Time in hours
time = np.array([0, 4, 8, 12, 16, 20])

# Temperature in °C
temperature = np.array([15, 18, 24, 29, 25, 19])

# Humidity in %
humidity = np.array([90, 85, 70, 60, 75, 88])

# Rainfall in mm
rainfall = np.array([2, 1.2, 0, 0, 0.5, 1.8])

# -------------------------------
# Step 2: Fit Quadratic Models (Iteration 2)
# -------------------------------
def fit_quadratic_model(x, y, label):
    coeffs = np.polyfit(x, y, 2)
    a, b, c = coeffs
    print(f"{label} Model: {a:.4f}t² + {b:.4f}t + {c:.4f}")
    return coeffs

print("\nDeveloped Quadratic Models:")
temp_coeffs = fit_quadratic_model(time, temperature, "Temperature")
hum_coeffs = fit_quadratic_model(time, humidity, "Humidity")
rain_coeffs = fit_quadratic_model(time, rainfall, "Rainfall")

# -------------------------------
# Step 3: Predictions (Iteration 2-3)
# -------------------------------
# Time from 0 to 24 hours
t_values = np.arange(0, 25, 1)

def predict(coeffs, t):
    return np.polyval(coeffs, t)

# Predict values
predicted_temp = np.clip(predict(temp_coeffs, t_values), -10, 50)
predicted_humidity = np.clip(predict(hum_coeffs, t_values), 0, 100)
predicted_rainfall = np.clip(predict(rain_coeffs, t_values), 0, None)

# -------------------------------
# Step 4: Print Forecast (Iteration 3)
# -------------------------------
print("\nPredicted Weather for 24 Hours:\n")
for t, temp, hum, rain in zip(t_values, predicted_temp, predicted_humidity, predicted_rainfall):
    print(f"At {t:02d}:00 hrs → Temp: {temp:.2f} °C | Humidity: {hum:.2f}% | Rainfall: {rain:.2f} mm")

# -------------------------------
# Step 5: Plotting (Iteration 4)
# -------------------------------
try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 10))

    # Temperature plot
    plt.subplot(3, 1, 1)
    plt.scatter(time, temperature, color='blue', label='Actual Temp')
    plt.plot(t_values, predicted_temp, 'r--', label='Predicted Temp')
    plt.ylabel('Temperature (°C)')
    plt.title('Temperature Forecast')
    plt.legend()
    plt.grid(True)

    # Humidity plot
    plt.subplot(3, 1, 2)
    plt.scatter(time, humidity, color='green', label='Actual Humidity')
    plt.plot(t_values, predicted_humidity, 'orange', linestyle='--', label='Predicted Humidity')
    plt.ylabel('Humidity (%)')
    plt.title('Humidity Forecast')
    plt.legend()
    plt.grid(True)

    # Rainfall plot
    plt.subplot(3, 1, 3)
    plt.scatter(time, rainfall, color='purple', label='Actual Rainfall')
    plt.plot(t_values, predicted_rainfall, 'black', linestyle='--', label='Predicted Rainfall')
    plt.xlabel('Time (Hours)')
    plt.ylabel('Rainfall (mm)')
    plt.title('Rainfall Forecast')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

except ImportError:
    print("\nNOTE: 'matplotlib' is not installed. Skipping graph display.")
    print("To install it, run: pip install matplotlib")
