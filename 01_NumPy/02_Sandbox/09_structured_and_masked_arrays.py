import numpy as np
import numpy.ma as ma

# 1. Structured Array (Mixed Data in one memory block)
data_type = [('Name', 'U10'), ('Age', 'i4'), ('Weight', 'f4')]
struct_arr = np.array([('Alex', 25, 68.5), ('Bob', 30, 75.2)], dtype=data_type)

# 2. Masked Array (Ignoring garbage without deleting)
raw_sensor_data = np.array([10.5, 12.1, -999.0, 11.8])
# Mask out -999.0 so it doesn't ruin our mean calculation
safe_data = ma.masked_equal(raw_sensor_data, -999.0)


# Raw data with multiple types of garbage values
raw_data = np.array([10.5, -999.0, 15.2, -888.0, 20.1, -999.0])

print("Original Data :", raw_data)

# Create a combined condition using Bitwise OR (|)
# Meaning: Mask the value IF it is -999.0 OR -888.0
condition = (raw_data == -999.0) | (raw_data == -888.0)

# Apply the mask based on the condition
safe_data = ma.masked_where(condition, raw_data)

print("Masked Data   :", safe_data)
print("Correct Mean  :", safe_data.mean())