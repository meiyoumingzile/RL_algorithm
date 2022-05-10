import ctypes
import os
from ctypes import cdll
import platform
# print(platform.architecture())
# ll=ctypes.cdll.LoadLibrary
print(os.path.exists("ygoforlinux.s"))
lib = cdll.LoadLibrary('ygoforlinux.so')
a= lib.testfun1(1,2)
print(a)
# print(lib.foo(1,1))