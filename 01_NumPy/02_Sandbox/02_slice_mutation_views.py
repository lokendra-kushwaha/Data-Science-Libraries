import numpy as np

print("** The Python List Way **")
py_list = [1, 2, 3, 4]
py_slice = py_list[0:2]  
py_slice[0] = 99 
print("Original Python List:", py_list)
print("Sliced Python List:", py_slice)

print("\n** The NumPy Way **")
np_array = np.array([1, 2, 3, 4])
np_slice = np_array[0:2]
np_slice[0] = 99
print("Original NumPy Array:", np_array)
print("Sliced NumPy Array:", np_slice)