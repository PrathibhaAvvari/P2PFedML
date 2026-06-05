import matplotlib.pyplot as plt

# X-axis: Number of faults
faults = [0, 1, 3, 5, 7, 9]

# Y-axis values (replace these with your actual data)
old_time = [10, 12, 18, 25, 30, 38]   # Old implementation times
new_time = [8, 10, 14, 20, 24, 29]    # New implementation times

# Create the plot
plt.plot(faults, old_time, marker='o', label='Old Implementation')
plt.plot(faults, new_time, marker='o', label='New Implementation')

# Labels and title
plt.xlabel("Number of Faults")
plt.ylabel("Time")
plt.title("Old vs New Implementation Performance")

# Show legend
plt.legend()

# Show grid for readability
plt.grid(True)

# Display the plot
plt.show()
