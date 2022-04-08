def christmasDay():
    # Prediction for Christmas Day Holiday
    # Libraries imports

    import numpy as np
    from functions.dataInput import load2013, load2014, load2015, load2016, load2017, load2018, load2019, temp2013, \
        temp2014, temp2015, temp2016, temp2017, temp2018, temp2019
    from functions.TransformAndCreationFunctions import daysOfTheWeekReferance, variablesCreation, setCreation
    from sklearn.neural_network import MLPRegressor
    from sklearn import metrics
    from sklearn.metrics import mean_absolute_percentage_error
    from matplotlib import pyplot as plt
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    # Holiday Datasets Lags

    # 6. ChristmasDay

    # Train Load lag inputs
    loadTrainDayTarget = load2016['2016-12-25']
    loadTrainDayMinus1 = load2016['2016-12-24']
    loadTrainDayMinus7 = load2016['2016-12-18']
    loadTrainDayMinus365 = load2015['2015-12-25']
    loadTrainDayMinus730 = load2014['2014-12-25']

    # Train temperature lag inputs
    temperTrainDayTarget = load2016['2016-12-25']
    temperTrainDayMinus1 = load2016['2016-12-24']
    temperTrainDayMinus7 = load2016['2016-12-18']
    temperTrainDayMinus365 = load2015['2015-12-25']
    temperTrainDayMinus730 = load2014['2014-12-25']

    ###############################################
    # Test lag days input
    loadTestDayTarget = load2019['2019-12-25']
    loadTestDayMinus1 = load2019['2019-12-24']
    loadTestDayMinus7 = load2019['2019-12-18']
    loadTestDayMinus365 = load2018['2018-12-25']
    loadTestDayMinus730 = load2017['2017-12-25']

    # test lag temperature input
    temperTestDayTarget = load2019['2019-12-25']
    temperTestDayMinus1 = load2019['2019-12-24']
    temperTestDayMinus7 = load2019['2019-12-18']
    temperTestDayMinus365 = load2018['2018-12-25']
    temperTestDayMinus730 = load2017['2017-12-25']
    ########################################################################################################################

    # Training Sets

    # creating  Variables for TrainSets

    (loadTarget, loadLag1, loadLag2, loadLag3, loadLag4, temperTarget, temperLag1, temperLag2, temperLag3, temperLag4) = \
        variablesCreation(loadTrainDayTarget, loadTrainDayMinus1, loadTrainDayMinus7, loadTrainDayMinus365,
                          loadTrainDayMinus730, temperTrainDayTarget, temperTrainDayMinus1, temperTrainDayMinus7,
                          temperTrainDayMinus365, temperTrainDayMinus730)

    # Creating Days Of the Week reference

    (weekDayLag1, weekDayLag2, weekDayLag3, weekDayLag4) = daysOfTheWeekReferance(loadTrainDayMinus1,
                                                                                  loadTrainDayMinus7,
                                                                                  loadTrainDayMinus365,
                                                                                  loadTrainDayMinus730)

    # Creating Train sets
    trainSetX, trainSetY = setCreation(loadTarget, loadLag1, weekDayLag1, loadLag2, weekDayLag2, loadLag3, weekDayLag3,
                                       loadLag4, weekDayLag4, temperTarget, temperLag1, temperLag2, temperLag3,
                                       temperLag4)

    ########################################################################################################################

    # Test Sets
    # creating  Variables for Test Sets

    (loadTarget, loadLag1, loadLag2, loadLag3, loadLag4, temperTarget, temperLag1, temperLag2, temperLag3, temperLag4) = \
        variablesCreation(loadTestDayTarget, loadTestDayMinus1, loadTestDayMinus7, loadTestDayMinus365,
                          loadTestDayMinus730, temperTestDayTarget, temperTestDayMinus1, temperTestDayMinus7,
                          temperTestDayMinus365, temperTestDayMinus730)

    # Creating Days Of the Week reference

    (weekDayLag1, weekDayLag2, weekDayLag3, weekDayLag4) = daysOfTheWeekReferance(loadTestDayMinus1, loadTestDayMinus7,
                                                                                  loadTestDayMinus365,
                                                                                  loadTestDayMinus730)

    # # Creating Test Sets

    testSetX, testSetY = setCreation(loadTarget, loadLag1, weekDayLag1, loadLag2, weekDayLag2, loadLag3, weekDayLag3,
                                     loadLag4, weekDayLag4, temperTarget, temperLag1, temperLag2, temperLag3,
                                     temperLag4)

    ###########################################################

    # Scaling
    scaler = StandardScaler()
    scaler_coeff = 1

    # Scaling Train Sets
    trainSetX = scaler.fit_transform(trainSetX) * scaler_coeff
    trainSetY = scaler.fit_transform(trainSetY) * scaler_coeff

    # Scaling Test sets

    testSetX = scaler.fit_transform(testSetX) * scaler_coeff
    testSetY = scaler.fit_transform(testSetY) * scaler_coeff

    ###########################################################

    # running the MLP regression
    MLPreg = MLPRegressor(hidden_layer_sizes=(18, 18, 18), activation="relu", solver='adam', random_state=1,
                          max_iter=2000, learning_rate='adaptive').fit(trainSetX, trainSetY.ravel())

    ###########################################################

    # predicting next day's load

    h_prediction = MLPreg.predict(testSetX)

    # revert scaling

    h_prediction = np.reshape(h_prediction, (-1, 1))
    h_prediction = scaler.inverse_transform(h_prediction / scaler_coeff)
    testSetY = scaler.inverse_transform(testSetY / scaler_coeff)

    # printing results

    print(h_prediction)
    print(testSetY)

    # calculating MAPE with sklearn
    mape = (mean_absolute_percentage_error(testSetY, h_prediction)) * 100
    maxError = metrics.max_error(testSetY, h_prediction)
    print('MAPE with sklearn is ' + str(mape))
    print('Model score by Metrics.R2 equals to ' + str(metrics.r2_score(testSetY, h_prediction)))
    print('Models max error is ' + str(maxError))

    # Plots
    plt.plot(h_prediction, label='Predicted Value')
    plt.plot(testSetY, label='Actual Value')
    plt.plot(loadTestDayMinus365, label=' 2018 ')
    plt.plot(loadTestDayMinus730, label='2017')
    plt.title('Christmas Day Holiday')
    plt.legend()
    plt.show()






