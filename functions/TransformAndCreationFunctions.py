# libraries imports
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
import math


# The function creates the load and temperature arrays for TrainSet

def variablesCreation(loadTarget, loadLag1, loadLag2, loadLag3, loadLag4, temperTarget, temperLag1, temperLag2, temperLag3, temperLag4):
    df = loadTarget
    x = df.to_numpy()
    df = loadLag1
    y = df.to_numpy()
    df = loadLag2
    z = df.to_numpy()
    df = loadLag3
    z1 = df.to_numpy()
    df = loadLag4
    z2 = df.to_numpy()
    df = temperTarget
    t1 = df.to_numpy()
    df = temperLag1
    t2 = df.to_numpy()
    df = temperLag2
    t3 = df.to_numpy()
    df = temperLag3
    t4 = df.to_numpy()
    df = temperLag4
    t5 = df.to_numpy()
    return x, y, z, z1, z2, t1, t2, t3, t4, t5


# Generation of the day of the week reference based on lags date

def daysOfTheWeekReferance(lag1, lag2, lag3, lag4):

    # D - 1
    temp1 = str(lag1.name)
    temp1 = temp1.split()
    lag1TiStamp = (time.mktime(datetime.datetime.strptime(temp1[0], "%Y-%m-%d").timetuple()))
    weekDayLag1 = datetime.datetime.fromtimestamp(lag1TiStamp).isoweekday()

    # D - 7
    temp1 = str(lag2.name)
    temp1 = temp1.split()
    lag2TiStamp = (time.mktime(datetime.datetime.strptime(temp1[0], "%Y-%m-%d").timetuple()))
    weekDayLag2 = datetime.datetime.fromtimestamp(lag2TiStamp).isoweekday()

    # D - 365
    temp1 = str(lag3.name)
    temp1 = temp1.split()
    lag3TiStamp = (time.mktime(datetime.datetime.strptime(temp1[0], "%Y-%m-%d").timetuple()))
    weekDayLag3 = datetime.datetime.fromtimestamp(lag3TiStamp).isoweekday()

    # D - 365x2
    temp1 = str(lag4.name)
    temp1 = temp1.split()
    lag4TiStamp = (time.mktime(datetime.datetime.strptime(temp1[0], "%Y-%m-%d").timetuple()))
    weekDayLag4 = datetime.datetime.fromtimestamp(lag4TiStamp).isoweekday()

    return weekDayLag1, weekDayLag2, weekDayLag3, weekDayLag4

# TrainSet and TestSet creation
def setCreation(target, lag1, weekDayLag1, lag2, weekDayLag2, lag3, weekDayLag3, lag4, weekDayLag4, temperTarget, temperLag1, temperLag2, temperLag3, temperLag4):

         #trainSetX
         maxLoad = ((max(lag3)) + (max(lag4))) / 2 # creating average max load values
         minLoad = ((min(lag3)) + (min(lag4))) / 2 # creating average min value
         data = []
         for i in range(0, 24):
              temp = [lag1[i], math.sin(weekDayLag1), lag2[i], math.sin(weekDayLag2), lag3[i], math.sin(weekDayLag3), lag4[i], math.sin(weekDayLag4), temperTarget[i], temperLag1[i], temperLag2[i], temperLag3[i], temperLag4[i], math.cos(i + 1), maxLoad, minLoad]
              data.append(temp)
         data1 = np.reshape(data, (24, len(temp)))
         #trainSetY
         data = []
         df = target
         for i in range(24):
              x = df[i]
              data.append(x)
         data2 = np.reshape(data, (-1, 1))
         return data1, data2


# plot function
def plots(prediction, actualValue):
    plt.plot(prediction, label='Prediction')
    plt.plot(actualValue, label='Actual Value')
    plt.xlabel('Hours')
    plt.ylabel('MW')
    plt.legend()
    plt.show()
