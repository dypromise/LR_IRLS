"""
cross valadation to chose a best l2_norm parameter 'lambda'.
"""
import IRLS_data_input
import IRLS_train
from  sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt

lambda_list = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]


def cv(K, X_, Y_):
    """

    :param K:
    :param X_:
    :param Y_:
    :return:
    """
    rs = ShuffleSplit(K, test_size=1.0 / K)
    accuarcy_list = []

    for lambda_ in lambda_list:
        acc_this_lambda = 0
        print "Training model with lambda: %f"%lambda_
        for train_index, test_index in rs.split(X_):
            LR = IRLS_train.model_Logistic_Regression()
            LR.train(X_[train_index], Y_[train_index], L2_norm_lambda=lambda_, L2_normalization=True)
            acc_this_lambda += LR.accuarcy_new_X(X_[test_index], Y_[test_index])

        acc_this_lambda /= K
        print "Accuarcy with lambda_ = %f is %f"%(lambda_,acc_this_lambda)
        accuarcy_list.append(acc_this_lambda)

    print "CV done. The accuarcy list is:"
    print accuarcy_list
    figure = plt.figure()
    plt.plot(range(len(lambda_list)), accuarcy_list)
    plt.xlabel('lambda index')
    plt.ylabel('accuarcy of 10-fold CV')
    plt.show()




if __name__ == '__main__':
    trainSet = IRLS_data_input.data_set('a9a')
    cv(10,trainSet.X_,trainSet.Y_)
