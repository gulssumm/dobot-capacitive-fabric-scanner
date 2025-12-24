from pydobot import *
import time

# A noktası
A = (0, 148.2, 11.9, -22.9, -0.4)
# B noktası
B = (0, 148.2, 104.3, -2.9, -0.4)

# Tek tek dene
magician.ptp(*A)
time.sleep(1)
magician.ptp(*B)

# Döngü
while True:
    magician.ptp(*A)
    time.sleep(1)
    magician.ptp(*B)
    time.sleep(1)
