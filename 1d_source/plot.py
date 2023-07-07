import utils
import matplotlib.pyplot as plt

confi = [0.014,1,75,"tanh","Glorot normal",1,1,1,1,1]
a = utils.create_model(confi)
p = utils.train_model(a)
utils.plot_3d(p)




