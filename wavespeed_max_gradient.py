#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 11:31:09 2019

@author: dung
"""

#%%
# Ziel : 
    # Der Druckverlauf in dem Zustand 5 : p5(t)
    # Das Zeitinterval für den Zustand 5: delta_t
# input : 
    # Drucksignals von den Drucksensoren: 4 Drucksensoren PCB und 1 Sensor nähe Endplatte
    # Die Positionen von den Drucksensoren
    # Die Temperatur, der Isentropenexponent, Der Druck P1
# output:
    # Die Machzahl Ms1
    # der Druck p5 = p1 * f(Ms1,gamma)
    # Das Zeitinterval delta_t = die Zeit zwichen 2 Mals Stoßendeckung nähe Endplatte (Sensor ch5)

#%% Input
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal as spsig
# Die Parameter von dem Anfangszustand
gamma = 1.4
p1 = 1 # bar
R = 8.314
T = 298.15
x = [-1202.46,-902.329,-602.67,-302.84,-3]  #Positionen von Drucksensoren 0 auf der Endwand
x_end = 0 # die Position von der Endplatte ist 0 festgelegt

# Read data von dem Experiment

data = pd.read_csv('pres_t061.csv',sep='\t',header=12)
data.columns = ['ch0','ch1','ch2','ch3','ch4','ch5','ch6','ch7']
chanel5 = data['ch5'].tolist()
time = range(0,len(data['ch0']),1) # Die globale Zeitraum für alle signale 

# Teilen die Signale in 2 Teilen (Hin- und Rückfahrt von der Stoßwelle)

pos_max5 = chanel5.index(max(chanel5))
Mp = pos_max5 - 2500
#print(Mp)
ch = np.array(data[['ch1','ch2','ch3','ch4','ch5']])
#%% Plotten

#plt.figure()
#plt.plot(time,ch[:0],label = 'ch1')
#plt.plot(time,ch[:1],label = 'ch2')
#plt.plot(time,ch[:2],label = 'ch3')
#plt.plot(time,ch[:3],label = 'ch4')
#plt.plot(time,ch[:4],label = 'ch5')
#plt.legend()
#plt.show()
#plt.figure()
#plt.plot(time,data['ch5'])
#plt.show()
#%% Die Funktion für Stoßerkennung
# finden maximum von Gradient von Signalen

def find_max_gradient(signal):
    t = np.linspace(0,len(signal)-1,len(signal))
    gr = np.gradient(signal,t)
    i = (np.where(gr == max(gr)))[0]
    index = i[0]
    return index

#%% Berechnen die Stoßgeschwindigkeiten

zeit = []
for j in range(0,5):
    cn = (ch[:,j])[0:Mp]
    zeit.append(find_max_gradient(cn))
v = [] # Stoßgeschwindigkeiten
for j in range(0,4):
    v.append((x[j+1]-x[j])*10000/(zeit[j+1]-zeit[j]))
pos = [] # Mittelpunkt von den sensor-Positionen
for k in range(0,4):
#    pos.append(x[k])
    pos.append((x[k]+x[k+1])/2)


# Extrapolieren die Geschwindigkeit am Rohrende

f = sp.interpolate.interp1d(pos,v,kind='linear',fill_value="extrapolate") 
v_end = f(x_end)

#%% Der Druck p5 und Das Zeitinterval im Zustand 5

M1 = v_end/(gamma*R*T/0.0289645)**0.5 # Machzahl von der einfallenden Stoßwelle

# Der Druck p5 Zustand 5

p5 = p1*((((3*gamma-1)/(gamma-1)*M1**2)-2)/(M1**2+2/(gamma-1)))*(((2*gamma/(gamma-1)*M1**2)-1)/((gamma+1)/(gamma-1)))

signal5 = (ch[:,4])[Mp:len(time)]

# Das Zeitinterval für den Zustand 5

delta_t = (find_max_gradient(signal5) + Mp - zeit[4])*10**(-7)
























