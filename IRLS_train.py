import numpy as np
import numpy.linalg as nl
import IRLS_data_input
import matplotlib.pyplot as plt


class model_Logistic_Regression(object):
    def __init__(self):

        # loop break threshold
        self.epsil = 1e-2

        # back search paras
        self.alpha = 0.3
        self.beta = 0.4

        # record the prediction and loss.
        self.iters = 1
        self.accuarcy_list = []
        self.lm_2_list = []

    def train(self, X_, Y_, L2_norm_lambda, L2_normalization=False):

        # training loop
        sample_num = len(X_)
        feature_num = len(X_[0])
        self.weights = np.random.randn(feature_num, 1) / 1000

        while (1):
            # computing prediction and accuarcy
            Y = self.sigmoid(np.dot(X_, self.weights))
            acc = self.compute_accuarcy(Y, Y_)
            self.accuarcy_list.append(acc)

            # computing Error of this iteration.
            # attention! the prediction Y may be 1, and cause log function doesn't work.
            Loss = -(np.dot(Y_.T, np.log(Y)) + np.dot((1 - Y_).T, np.log(1.0 - Y)))

            # computing gradients.
            grad = np.dot(X_.T, Y - Y_)
            if (L2_normalization):
                grad += L2_norm_lambda * self.weights

            # computing Hession matrix.
            R = np.diag(np.reshape(Y * (1 - Y), sample_num))
            Hession = np.dot(np.dot(X_.T, R), X_)
            if (L2_normalization):
                Hession += L2_norm_lambda * np.diag(np.ones(feature_num))

            # computing descent direction.If no normalization term ,H may be singular.
            delta_x = -np.dot(nl.inv(Hession), grad)

            # weather to break the loop
            lm_2 = -np.dot(grad.T, delta_x)
            self.lm_2_list.append(lm_2.reshape(1))
            print "Iteration: %d, Loss: %f, lambda^2: %f" % (self.iters, Loss, lm_2)
            if (lm_2 < self.epsil):
                break

            # back tracking search for learning rate and updating weights.
            self.weights = self.back_track_search(Loss, grad, delta_x, self.weights, X_, Y_)
            self.iters += 1

            # self.plot_iter_loss()

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def Loss_of_current_weights(self, weights, X_, Y_):
        """

        :param weights: current weights of LR.
        :return: The Loss of current LR with given weights. This need training set X_,Y_.
        """
        Y = self.sigmoid(np.dot(X_, weights))
        return -(np.dot(Y_.T, np.log(Y)) + np.dot((1 - Y_.T), np.log(1 - Y)))

    def back_track_search(self, fx_val, grad, delta_x, x, X_, Y_):
        """

        :param fx: current value of target function of current training loop.
        :param grad: the gradients of target function of current training loop.
        :param delta_x: the dscent direction of this training loop.
        :param x: the old var to opt.
        :param func: OPt func.
        :return: new X.
        """
        t = 1.0
        x_new = x + t * delta_x
        while (self.Loss_of_current_weights(x_new, X_, Y_) > fx_val + self.alpha * t * np.dot(grad.T, delta_x)):
            t *= self.beta
            x_new = x + t * delta_x
        return x_new

    def compute_accuarcy(self, Y, Y_):
        """

        :param Y:predictions
        :param Y_: targets
        :return: accuarcy
        """
        acc_sum = 0.0
        for i in xrange(len(Y)):
            if (Y[i] < 0.5):
                acc_sum += np.fabs(0 - Y_[i])
            else:
                acc_sum += np.fabs(1 - Y_[i])
        return acc_sum / len(Y)

    def plot_iter_loss(self):
        figure = plt.figure()
        x = range(self.iters)
        plt.subplot(2, 1, 1)
        plt.plot(x, self.accuarcy_list)
        plt.xlabel('iterations')
        plt.ylabel('accuarcy')
        plt.subplot(2, 1, 2)
        plt.plot(x, self.lm_2_list)
        plt.xlabel('iterations')
        plt.ylabel('lmbda^2: stop iter threshold')
        plt.show()

    def accuarcy_new_X(self, X_, Y_):
        """
        use new X_ and trained model weights to caculate prediction,
        and compute accuarcy compare to real traget Y_.
        :param X_: new input X_
        :param Y_: new target Y_
        :return: accuarcy
        """
        Y = self.sigmoid(np.dot(X_, self.weights))
        return self.compute_accuarcy(Y, Y_)


if __name__ == '__main__':
    TrainSet = IRLS_data_input.data_set('a9a')
    TestSet = IRLS_data_input.data_set('a9a.t')

    LR = model_Logistic_Regression()
    print "training..."
    LR.train(TrainSet.X_, TrainSet.Y_, L2_norm_lambda=0.003, L2_normalization=True)
    print "training done."
    acc = LR.accuarcy_new_X(TestSet.X_, TestSet.Y_)
    print "Accuarcy in train set is: %f" % LR.accuarcy_list[-1]
    print "Accuarcy in test set is: %f" % acc
    LR.plot_iter_loss()
