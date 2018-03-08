# @Date:   12-10-2017
# @Last modified time: 12-10-2017
# @License: GNU Public License v3


#from socket import *
from threading import *
#from thread import *

#screenLock = Semaphore(value=1)

def calc(i):
    return i+1

i = 0
for i in range (0,255):
    t = Thread(target=calc,args=i)
    receive= t.start()

print(receive)
