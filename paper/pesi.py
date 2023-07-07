a = numpy.load('23.03.28_reformulation/weights_lambda_5.npy', allow_pickle=True)
t = a[0]
y = a[1:]
y[:,-1] #sono i pesi all'istante finale
