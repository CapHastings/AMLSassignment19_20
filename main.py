from A1 import a1
from A2 import a2
from B1 import b1
from B2 import b2

# ======================================================================================================================
# Data preprocessing
data_train_a1, data_val_a1, data_test_a1 = a1.data_preprocessing()
data_train_a2, data_val_a2, data_test_a2 = a2.data_preprocessing()
data_train_b1, data_val_b1, data_test_b1 = b1.data_preprocessing()
data_train_b2, data_val_b2, data_test_b2 = b2.data_preprocessing()
print('All data loaded.')

# ======================================================================================================================
# Task A1
model_A1 = a1.A1()
acc_A1_train = model_A1.train(data_train_a1, data_val_a1)
acc_A1_test = model_A1.test(data_test_a1)
print('Task A1 completed.')
del model_A1, data_train_a1, data_val_a1, data_test_a1

# ======================================================================================================================
# Task A2
model_A2 = a2.A2()
acc_A2_train = model_A2.train(data_train_a2, data_val_a2,
                              load_model=True)  # setting load_model=False takes 30 minutes to train a CNN from scratch,
                                                 # whereas setting load_model=True loads a pre-trained model.
acc_A2_test = model_A2.test(data_test_a2)
print('Task A2 completed.')
del model_A2, data_train_a2, data_val_a2, data_test_a2

# ======================================================================================================================
# Task B1
model_B1 = b1.B1()
acc_B1_train = model_B1.train(data_train_b1, data_val_b1)
acc_B1_test = model_B1.test(data_test_b1)
print('Task B1 completed.')
del model_B1, data_train_b1, data_val_b1, data_test_b1

# ======================================================================================================================
# Task B2
model_B2 = b2.B2()
acc_B2_train = model_B2.train(data_train_b2, data_val_b2)
acc_B2_test = model_B2.test(data_test_b2)
print('Task B2 completed.')
del model_B2, data_train_b2, data_val_b2, data_test_b2

# ======================================================================================================================
print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
                                                        acc_A2_train, acc_A2_test,
                                                        acc_B1_train, acc_B1_test,
                                                        acc_B2_train, acc_B2_test
                                                        ))
