# OpenAI Gym original environments

## Requirement 
- python3.6
- OpenAI Gym

## Building OpenAI Gym from source code

```
git clone https://github.com/openai/gym
cd gym
pip install -e .
```

## Environment

```python
import gym
import ss2d

env = gym.Make('Simple-v0')
env.reset()

for _ i in range(1000):
    env.render()
    observation, reward, done, _ = env.step(env.action_space.sample()) # take a random action

    if done:
        env.reset()
```

It should look someting like this test

![demo](https://github.com/tsuchiya-i/SS2D/blob/main/navigation_sample.gif)


