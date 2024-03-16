# PasswordBruteForce-PythonBruteForce

Authors: [Marek von Rogall](https://github.com/marekvonrogall) & [Sprudello](https://github.com/sprudello)

## Purpose:
This application tries to find out a password through trial and error. The condition for this is that the length of the password is known.
This work was a leisure project.

## Configuration &Usage:
The program uses [pycuda](https://pypi.org/project/pycuda/) to perform fast computation on the GPU's threads. However, the implementation of pycuda requires a CUDA compatible NVIDIA graphics card.
The prerequisite is that the CUDA toolkit is installed on your system. You can get the CUDA Toolkit from NVIDIA here: [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)

Before the project can be executed, a few settings must be made. The character set that should be used to find out the password can be adjusted in the program code as well as the rate you want the threads to display the current password they generated. It is also necessary to set the threads to be used per block.

### Changing the character set: [here](https://github.com/marekvonrogall/PyCUDA_PythonBruteForcer/blob/50c1f4cc0113538ad4150c02d79977363ebfece9/PythonBruteForce/PythonBruteForce.py#L124)
```py
characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
```
The character set can be adjusted as desired. The program will only generate passwords with these characters to find out the password you entered.

_Note: The more characters the character set contains, the more possible combinations the password unknown to the program has. This means that the process may take longer._

### Changing the update rate: [here](https://github.com/marekvonrogall/PyCUDA_PythonBruteForcer/blob/50c1f4cc0113538ad4150c02d79977363ebfece9/PythonBruteForce/PythonBruteForce.py#L144)
```py
update_rate = 0
```

### Adjusting the threads used per block: [here](https://github.com/marekvonrogall/PyCUDA_PythonBruteForcer/blob/50c1f4cc0113538ad4150c02d79977363ebfece9/PythonBruteForce/PythonBruteForce.py#L145)
```py
threads_per_block = 1024
```
