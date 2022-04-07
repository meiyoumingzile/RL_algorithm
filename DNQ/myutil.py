import os

def initInfo(s,f="mod.txt"):
    if os.path.exists(f):
        os.remove(f)
    print(s)
    with open(f, 'a') as f:
        f.write(s + "\n")
def wInfo(s,f="mod.txt"):
    print(s)
    with open(f, 'a') as f:
        f.write(s + "\n")