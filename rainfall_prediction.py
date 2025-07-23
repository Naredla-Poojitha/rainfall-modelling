
import numpy as np

# Step 1: Input Data (hourly rainfall)
time = np.array([19, 20, 21, 22, 23, 24, 25, 26])  # Hours since midnight of previous day
rainfall = np.array([0.0, 0.0, 0.0, 0.01, 0.03, 0.01, 0.0, 0.02])  # Rainfall in mm

# Step 2: Fit the quadratic model R(t) = a*t^2 + b*t + c
coefficients = np.polyfit(time, rainfall, 2)
a, b, c = coefficients

print(f"\nDeveloped Quadratic Model:\nR(t) = {a:.6f}t² + {b:.4f}t + {c:.4f}\n")

# Step 3: Predict rainfall for every hour from 19h to 30h (~6 AM next day)
t_values = np.arange(19, 31, 1)
predicted_rain = a * t_values**2 + b * t_values + c

print("Predicted Rainfall (mm) for Next Hours:\n")
for t, amt in zip(t_values, predicted_rain):
    hour = t % 24
    day = 'today' if t < 24 else 'tomorrow'
    print(f"At {hour:02d}:00 hrs {day} -> {amt:.3f} mm")

# Step 4: Plot (if matplotlib is installed)
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.scatter(time, rainfall, color='blue', label='Recorded Rainfall', zorder=5)
    plt.plot(t_values, predicted_rain, 'r--', label='Quadratic Model Prediction')
    plt.title('Rainfall Prediction using Quadratic Model')
    plt.xlabel('Time (Hours since previous midnight)')
    plt.ylabel('Rainfall (mm)')
    plt.xticks(t_values)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
except ImportError:
    print("matplotlib not installed; skipping plot.")
