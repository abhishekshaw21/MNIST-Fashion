# Name     : Abhishek Shaw
# Description: Multi-Layer Perceptron for Cloth Classification
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import sys
import tensorflow as tf
from sklearn.model_selection import train_test_split
import platform

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
total_class = len(class_names)
# Parameters
training_epochs = 100
learning_rate = 0.001
batch_size = 50
patience = int(np.sqrt(training_epochs))


def softMax(layer3):
    meanLayer = tf.reduce_max(layer3)
    sub = tf.subtract(layer3, meanLayer)
    expLayer = tf.exp(sub)
    sum1 = tf.reduce_sum(expLayer)
    output = tf.divide(expLayer, sum1)
    return output


def saveWeight(fileName, weights):
    pass


def loadWeight(fileName, weights):
    pass


def reluActivation(a):
    return tf.maximum(a, 0)


def test():
    print("+" * 80)
    print("Testing Phase")
    print("+" * 80)
    import mnist_reader
    X_test, y_test = mnist_reader.load_mnist('../data/fashion', kind='t10k')
    X_test = -0.5 + (X_test / 255.0)
    y_test = np.eye(total_class)[y_test]
    print("Length of Training Data:", len(X_test))
    tf.compat.v1.reset_default_graph()
    saver = tf.compat.v1.train.import_meta_graph("weight/model.ckpt.meta")
    x = tf.compat.v1.placeholder(tf.float64, [None, 784])
    y_ = tf.compat.v1.placeholder(tf.float64, [None, 10])
    with tf.compat.v1.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint("weight/"))
        graph = tf.compat.v1.get_default_graph()
        w1 = graph.get_tensor_by_name("w1:0")
        b1 = graph.get_tensor_by_name("b1:0")
        w2 = graph.get_tensor_by_name("w2:0")
        b2 = graph.get_tensor_by_name("b2:0")
        w3 = graph.get_tensor_by_name("w3:0")
        b3 = graph.get_tensor_by_name("b3:0")
        w4 = graph.get_tensor_by_name("w4:0")
        b4 = graph.get_tensor_by_name("b4:0")
        h1 = tf.add(tf.matmul(x, w1), b1)
        h1 = reluActivation(h1)
        h2 = tf.add(tf.matmul(h1, w2), b2)
        h2 = reluActivation(h2)
        h3 = tf.add(tf.matmul(h2, w3), b3)
        h3 = reluActivation(h3)
        predicted = tf.add(tf.matmul(h3, w4), b4)
        predicted = tf.argmax(predicted, 1)
        actual = tf.argmax(y_, 1)
        correct_prediction = tf.equal(predicted, actual)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
        test_accuracy = sess.run(accuracy, feed_dict={x: X_test, y_: y_test})
        print("Test Accuracy:", round(test_accuracy * 100, 3), "%")


def train(hidden_unit_size=64):
    global learning_rate
    global patience
    print("+" * 80)
    print("Training Phase")
    print("+" * 80)
    import mnist_reader
    X_train, y_train = mnist_reader.load_mnist('../data/fashion', kind='train')
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=(1 / 6), random_state=48)
    # Normalize input
    X_train = -0.5 + (X_train / 255.0)
    X_valid = -0.5 + (X_valid / 255.0)
    # One Hot Encoding for Training and Validation Set
    y_train = np.eye(total_class)[y_train]
    y_valid = np.eye(total_class)[y_valid]
    print("Length of Training Data:", len(X_train))
    print("Length of Validation Data", len(X_valid))
    # Logits
    x = tf.compat.v1.placeholder(tf.float64, [None, 784])
    w1 = tf.Variable(tf.random.normal(
        [784, hidden_unit_size], mean=0.0, stddev=np.math.sqrt(2 / (hidden_unit_size)), dtype=tf.float64), name="w1")
    b1 = tf.Variable(tf.zeros([hidden_unit_size],
                              dtype=tf.float64),  name="b1")
    layer1 = tf.add(tf.matmul(x, w1), b1)
    layer1 = reluActivation(layer1)
    w2 = tf.Variable(tf.random.normal(
        [hidden_unit_size, hidden_unit_size], mean=0.0, stddev=np.math.sqrt(2 / (hidden_unit_size)), dtype=tf.float64), name="w2")
    b2 = tf.Variable(tf.zeros(
        [hidden_unit_size], dtype=tf.float64), name="b2")
    layer2 = tf.add(tf.matmul(layer1, w2), b2)
    layer2 = reluActivation(layer2)
    w3 = tf.Variable(tf.random.normal(
        [hidden_unit_size, hidden_unit_size],  mean=0.0, stddev=np.math.sqrt(2 / (hidden_unit_size)), dtype=tf.float64), name="w3")
    b3 = tf.Variable(tf.zeros([hidden_unit_size], dtype=tf.float64), name="b3")
    layer3 = tf.add(tf.matmul(layer2, w3), b3)
    layer3 = reluActivation(layer3)
    w4 = tf.Variable(tf.random.normal(
        [hidden_unit_size, 10],  mean=0.0, stddev=np.math.sqrt(2 / (hidden_unit_size)), dtype=tf.float64), name="w4")
    b4 = tf.Variable(tf.zeros([10], dtype=tf.float64), name="b4")
    layer4 = tf.add(tf.matmul(layer3, w4), b4)
    y_ = tf.compat.v1.placeholder(tf.float64, [None, 10])
    output = softMax(layer4)
    # loss calculation
    avgerageLoss = tf.reduce_mean(tf.compat.v1.losses.softmax_cross_entropy(y_, layer4)) + 0.0002 * (tf.reduce_sum(tf.multiply(
        w1, w1)) + tf.reduce_sum(tf.multiply(w2, w2)) + tf.reduce_sum(tf.multiply(w3, w3)) + tf.reduce_sum(tf.multiply(w4, w4)))
    # Minimize Cost
    optimizer_adam = tf.compat.v1.train.AdamOptimizer(
        learning_rate=learning_rate).minimize(avgerageLoss)
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))
    validationLoss = tf.reduce_mean(
        tf.compat.v1.losses.softmax_cross_entropy(y_, layer4))
    # Initialize Variable
    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver()
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        best_val = 0
        best_val_e = -1
        for e in range(training_epochs):
            print("Epoch", e + 1)
            epoch_loss = 0
            for batch_index in range(0, X_train.shape[0], batch_size):
                miniBatch_X = X_train[batch_index:batch_index + batch_size, :]
                miniBatch_Y = y_train[batch_index:batch_index + batch_size, :]
                _, batch_loss = sess.run([optimizer_adam, avgerageLoss], feed_dict={
                    x: miniBatch_X, y_: miniBatch_Y})
                epoch_loss += batch_loss
            epoch_loss = epoch_loss / (X_train.shape[0] / batch_size)
            print("Train_loss: ", epoch_loss, end=" ")
            trainAccuracy = sess.run(accuracy, feed_dict={
                x: X_train, y_: y_train})
            print("Train_accuracy:", round(float(trainAccuracy * 100), 3), "%")
            validLoss, validAccuracy = sess.run([validationLoss, accuracy], feed_dict={
                x: X_valid, y_: y_valid})
            print("Validation loss:", validLoss, end=" ")
            print("Validation Accuracy:", round(validAccuracy * 100, 3), "%")
            if(best_val < validAccuracy):
                best_val = validAccuracy
                best_val_e = e
                save_path = saver.save(sess, "weight/model.ckpt")
            elif(best_val_e + patience <= e):
                print("Early Stopping.\nUsing Epoch", best_val_e + 1)
                print("Validation Accuracy", round(best_val * 100, 3), "%")
                break


def logisticRegressionCustom(layer):
    try:
        layer = int(layer)
    except Exception as e:
        print(e)
    if(layer > 3 or layer < 1):
        print("Invalid Layer Number")
        sys.exit(0)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    import mnist_reader
    X_train, y_train = mnist_reader.load_mnist('../data/fashion', kind='train')
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=(1 / 6), random_state=48)
    X_test, y_test = mnist_reader.load_mnist('../data/fashion', kind='t10k')
    # Normalize input
    X_train = -0.5 + (X_train / 255.0)
    X_valid = -0.5 + (X_valid / 255.0)
    X_test = -0.5 + (X_test / 255.0)
    # Logits
    tf.compat.v1.reset_default_graph()
    saver = tf.compat.v1.train.import_meta_graph("weight/model.ckpt.meta")
    x = tf.compat.v1.placeholder(tf.float64, [None, 784])
    y_ = tf.compat.v1.placeholder(tf.float64, [None, 10])
    with tf.compat.v1.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint("weight/"))
        graph = tf.compat.v1.get_default_graph()
        w1 = graph.get_tensor_by_name("w1:0")
        b1 = graph.get_tensor_by_name("b1:0")
        w2 = graph.get_tensor_by_name("w2:0")
        b2 = graph.get_tensor_by_name("b2:0")
        w3 = graph.get_tensor_by_name("w3:0")
        b3 = graph.get_tensor_by_name("b3:0")
        w4 = graph.get_tensor_by_name("w4:0")
        b4 = graph.get_tensor_by_name("b4:0")
        h1 = tf.add(tf.matmul(x, w1), b1)
        h1 = reluActivation(h1)
        h2 = tf.add(tf.matmul(h1, w2), b2)
        h2 = reluActivation(h2)
        h3 = tf.add(tf.matmul(h2, w3), b3)
        h3 = reluActivation(h3)
        if(layer == 1):
            logisticTrain = sess.run(h1, feed_dict={x: X_train})
            logisticTest = sess.run(h1, feed_dict={x: X_test})
        elif(layer == 2):
            logisticTrain = sess.run(h2, feed_dict={x: X_train})
            logisticTest = sess.run(h2, feed_dict={x: X_test})
        elif(layer == 3):
            logisticTrain = sess.run(h3, feed_dict={x: X_train})
            logisticTest = sess.run(h3, feed_dict={x: X_test})
        logisticregression = LogisticRegression(max_iter=1000)
        logisticregression.fit(logisticTrain, y_train)
        test_prediction = logisticregression.predict(logisticTest)
        print("Logistic Regression Output")
        print('Using layer', layer, 'as input')
        print("*" * 20)
        print('Accuracy of logistic regression classifier on test set:',
              logisticregression.score(logisticTest, y_test) * 100, "%")


def main():
    print("*" * 80)
    print("TensorFlow Version\nRunning: ",
          tf.__version__ + "\nDeveloper Version: 1.14.0")
    print("Python Version\nRunning: ", platform.python_version() +
          "\nDeveloper Version: 3.6.10")
    print("*" * 80)
    if(len(sys.argv) != 2):
        print("Invalid number of command line arrgument provided.Use")
        print("--train :to Train the model")
        print("--test  :to Test the model")
        print("--layer=<layer_number>")
        sys.exit()
    if(sys.argv[1] == "--train"):
        train(48)
        sys.exit()
    elif(sys.argv[1] == "--test"):
        test()
        sys.exit()
    elif("--layer=" in sys.argv[1]):
        logisticRegressionCustom(sys.argv[1][-1])
    else:
        print("Invalid argument provided.Use")
        print("--train :to Train the model")
        print("--test  :to Test the model")
        print("--layer=<layer_number>")

if __name__ == "__main__":
    main()
