# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 13:29:32 2025

@author: Aidata
"""

from pydobot import Dobot
import time

PORT = "COM3"   # Aygıt Yöneticisi'nden kontrol et

device = Dobot(port=PORT, verbose=True)

# A noktası
A = (148.2, 11.9, -22.9, -0.4)
device.move_to(*A, wait=True)

time.sleep(1)
device.close()
