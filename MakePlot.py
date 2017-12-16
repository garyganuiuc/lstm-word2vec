import numpy as np
import matplotlib.pyplot as plt

#web
#loss
y_simple_loss100 = np.load("tr_loss_simple.npy")
x_simple_loss100 = list(range(len(y_simple_loss100)))

fig = plt.figure()
plt.plot(x_simple_loss100, y_simple_loss100, label = 'loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('epoch vs. loss for Reuters data')
plt.show()
plt.savefig('p1.png')

#train acc
y_simple_tr_acc100 = np.load("tr_acc_simple.npy")
x_simple_tr_acc100 = list(range(len(y_simple_tr_acc100)))

fig2 = plt.figure()
plt.plot(x_simple_tr_acc100, y_simple_tr_acc100, label = 'train accuracy')
plt.xlabel('epoch')
plt.ylabel('training accuracy')
plt.title('epoch vs. training accuracy for Reuters data')
plt.show()
plt.savefig('p2.png')

#test acc
y_simple_te_acc100 = np.load("tst_acc_simple.npy")
x_simple_te_acc100 = list(range(len(y_simple_te_acc100)))

fig3 = plt.figure()
plt.plot(x_simple_te_acc100, [ 450*x for x in y_simple_te_acc100], label = 'test accuracy')
plt.xlabel('epoch')
plt.ylabel('testing accuracy')
plt.title('epoch vs. testing accuracy for Reuters data')
plt.show()
plt.savefig('p3.png')

#local
#train loss
y_simple_loss100 = np.load("train_loss_simple_local.npy")
x_simple_loss100 = list(range(len(y_simple_loss100)))

fig4 = plt.figure()
plt.plot(x_simple_loss100, y_simple_loss100, label = 'loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('epoch vs. loss for 100 epoch')
plt.show()

#train acc
y_simple_tr_acc100 = np.load("train_acc_simple_local.npy")
x_simple_tr_acc100 = list(range(len(y_simple_tr_acc100)))

fig5 = plt.figure()
plt.plot(x_simple_tr_acc100, y_simple_tr_acc100, label = 'train accuracy')
plt.xlabel('epoch')
plt.ylabel('training accuracy')
plt.title('epoch vs. training accuracy for 100 epoch')
plt.show()

#test acc
y_simple_te_acc100 = np.load("test_acc_simple_local.npy")
x_simple_te_acc100 = list(range(len(y_simple_te_acc100)))

fig6 = plt.figure()
plt.plot(x_simple_te_acc100, y_simple_te_acc100, label = 'test accuracy')
plt.xlabel('epoch')
plt.ylabel('testing accuracy')
plt.title('epoch vs. testing accuracy for 100 epoch')
plt.show()
