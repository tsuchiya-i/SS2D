# Simple robot Simulator 2D
## Python
Python version 3.6 or higher is recommended.
## Requirement 
- Python3.6
- OpenAI Gym
- OpenCV
- Pillow
- [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2)
- tensorflow
- Keras-rl2

## Building RVO2 from source code
Use the following command to install RVO2. ([Detail](https://github.com/sybrenstuvel/Python-RVO2))
```
$ pip3 install Cython
$ git clone https://github.com/sybrenstuvel/Python-RVO2.git
$ cd Python-RVO2
$ python3 setup.py build
$ python3 setup.py install
```
If you get a permission error, run this command.
```
$ sudo python3 setup.py install
```

## Building OpenAI Gym from source code
```
git clone https://github.com/openai/gym
cd gym
pip install -e .
```

## Installing TensorFlow
```
$ pip3 install tensorflow
```
If you want to use a GPU, please check these versions of TensorFlow, CUDA, and NVIDIA-Drivers and install them.([TensorFlow & CUDA](https://www.tensorflow.org/install/source?hl=ja#tested_build_configurations))([CUDA & Driver](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html))

## Installing SS2D
```
$ git clone https://github.com/tsuchiya-i/SS2D.git
$ cd SS2D
$ pip3 install -e .
```

## Note
If you get this error
```
ModuleNotFoundError: No module named 'tkinter'
```
Then run this command
```
sudo apt-get install python3-tk
```

If you get this error
```
ModuleNotFoundError: No module named 'tkinter'
```
Then run this command
```
sudo apt-get install python3-tk
```


## Demo
![demo](https://github.com/tsuchiya-i/SS2D/blob/main/navigation_sample.gif)


