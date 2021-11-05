# OpenAI Gym original environments

## Requirement 
- python3.6
- OpenAI Gym
- OpenCV
- Python-RVO2

## Building OpenAI Gym from source code

```
git clone https://github.com/openai/gym
cd gym
pip install -e .
```
## Building RVO2 from source code

```
git clone https://github.com/sybrenstuvel/Python-RVO2.git
cd Python-RVO2
pip install -r requirements.txt
python setup.py build
python setup.py install
```
## Install OpenCV

```
pip install opencv-python
```

It should look someting like this test

![demo](https://github.com/tsuchiya-i/SS2D/blob/main/navigation_sample.gif)


