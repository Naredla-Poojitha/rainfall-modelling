
import numpy as np

# Step 1: Input Data (humidity readings hourly)
time = np.array([0, 1, 2, 3, 4, 5, 6, 7])  # Time in hours since midnight
humidity = np.array([78, 83, 80, 81, 81, 84, 81, 78])  # Humidity in %

# Step 2: Fit the quadratic model H(t) = a*t^2 + b*t + c
coefficients = np.polyfit(time, humidity, 2)
a, b, c = coefficients

print(f"\nDeveloped Quadratic Model:\nH(t) = {a:.4f}t² + {b:.4f}t + {c:.4f}\n")

# Step 3: Predict humidity for every hour from 0 to 24
t_values = np.arange(0, 25, 1)
predicted_humidity = a * t_values**2 + b * t_values + c

print("Predicted Humidity (%) for 24 Hours:\n")
for t, hum in zip(t_values, predicted_humidity):
    print(f"At {t:02d}:00 hrs -> {hum:.2f} %")

# Step 4: Plot (if matplotlib is installed)
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.scatter(time, humidity, color='blue', label='Original Humidity Data', zorder=5)
    plt.plot(t_values, predicted_humidity, 'r--', label='Quadratic Model Fit')
    plt.title('Humidity Prediction using Quadratic Model')
    plt.xlabel('Time (Hours since midnight)')
    plt.ylabel('Humidity (%)')
    plt.xticks(np.arange(0, 25, 2))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
except ImportError:
    print("matplotlib not installed; skipping plot.")
