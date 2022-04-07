import sys

import gym
import pynvml
pynvml.nvmlInit()
def checkWD():
    gpu_device1 = pynvml.nvmlDeviceGetHandleByIndex(0)
    temperature1 = pynvml.nvmlDeviceGetTemperature(gpu_device1, pynvml.NVML_TEMPERATURE_GPU)
    gpu_device2 = pynvml.nvmlDeviceGetHandleByIndex(1)
    temperature2 = pynvml.nvmlDeviceGetTemperature(gpu_device2, pynvml.NVML_TEMPERATURE_GPU)
    if temperature1>76 or temperature2>76:
        print(str(temperature1)+"  "+str(temperature2)+" exit!!!")
        sys.exit(0)

def resetPic(pic):
    return pic
def setReward(env,state):#设置奖励函数函数
    x, x_dot, theta, theta_dot = state
    r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
    r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
    return r1 + r2