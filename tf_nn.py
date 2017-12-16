def nn(train_x,train_y,test_x,test_y,LR, keep_prob1, num_epochs):
    import numpy as np
    import h5py
    import time

    import tensorflow as tf
    L_Y_train = train_y.shape[0]
    L_Y_test = test_y.shape[0]
    #number of hidden units
    H = 250
    #number of epochs
    #num_epochs = 200
    batch_size = 100
    #learning rate
    #LR = .1

    #model is trained using stochastic gradient descent

    #available devices: gpu:0 (default), cpu:0, cpu:1, ..., cpu:15
    #by default, model is trained on gpu:0 (the best available device)
    #keep_prob1 = 0.9

    time1 = time.time()
    with tf.device('/gpu:0'):
    #with tf.device('/cpu:0'):
        n_feature = train_x.shape[1] 
        x = tf.placeholder(tf.float32, shape=[None, n_feature])
        y_ = tf.placeholder(tf.float32, shape=[None, 2])
        keep_prob = tf.placeholder(tf.float32)
        #single layer neural network
        W1 = tf.Variable(tf.random_normal([n_feature,H], stddev=1.0/np.sqrt(n_feature) ) )
        b1 = tf.Variable( tf.zeros([H] ) )
        W2 = tf.Variable( tf.random_normal([H, 2], stddev=1.0/np.sqrt(H) )  )
        
        b2 = tf.Variable( tf.zeros([2]) )

        h1 = tf.nn.relu( tf.matmul(x, W1) + b1 )
        h1_drop = tf.nn.dropout(h1, keep_prob)
        y = tf.nn.softmax(tf.matmul(h1_drop,W2) + b2)

        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        train_step = tf.train.GradientDescentOptimizer(LR).minimize(cross_entropy)
    
        sess = tf.Session()

        sess.run(tf.global_variables_initializer())     
        for epochs in range(num_epochs):
            for i in range(0, L_Y_train, batch_size):
                #train_step.run(feed_dict={x: x_train[i:i+batch_size,:], y_: y_train2[i:i+batch_size,:]  })
                _, train_accuracy, loss = sess.run([train_step, accuracy, cross_entropy],feed_dict={x: train_x.tocsr()[i:min(i+batch_size,L_Y_train),:].toarray(), y_: train_y[i:min(i+batch_size,L_Y_train),:],keep_prob:keep_prob1})
                # _ = sess.run([train_step],feed_dict={x: x_train[i:i+batch_size,:], y_: y_train2[i:i+batch_size,:]})
            test_accuracy = 0.0
            for i in range(0,test_y.shape[0],batch_size):
                test_accuracy_tmp = sess.run([accuracy],feed_dict={x: test_x.tocsr()[i:min(i+batch_size,L_Y_test),:].toarray(), y_: test_y[i:min(i+batch_size,L_Y_test),:],keep_prob:1.0})
                test_accuracy += test_accuracy_tmp[0]*(min(i+batch_size,L_Y_test)-i)
            test_accuracy = test_accuracy/L_Y_test
            print("Test Accuracy: %f" % test_accuracy)


            print("epoch:%d accuracy:%f,loss:%f" % (epochs,train_accuracy,loss))

        #test_accuracy = sess.run([accuracy],feed_dict={x: test_x, y_: test_y})
        #print("Test Accuracy: %f" % test_accuracy[0])



        time2 = time.time()
        total_time = time2 - time1
        print("Time: %f" % total_time)
