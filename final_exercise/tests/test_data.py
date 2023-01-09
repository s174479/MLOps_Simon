from data import mnist
from tests import _PATH_DATA
import pytest
import os.path
@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data_train_size():
    train, test = mnist()
    assert len(train) == 25000, "Train dataset did not have the correct number of samples"

def test_data_test_size():    
    train, test = mnist()
    assert len(test) == 5000, "Test dataset did not have the correct number of samples"

train, test = mnist()
for i in range(len(train)):
    assert len(train[i][0]) == 28, "Images in train does not have the correct amount of rows"
    assert len(train[i][0][0]) == 28, "Images in train does not have the correct amount of columns"
    assert train[i][1] >= 0 and  train[i][1] <= 9, "Labels in train are not all between 0 and 9"
for i in range(len(test)):
    assert len(test[i][0]) == 28, "Images in test does not have the correct amount of rows"
    assert len(test[i][0][0]) == 28, "Images in test does not have the correct amount of columns"
    assert test[i][1] >= 0 and  test[i][1] <= 9, "Labels in test are not all between 0 and 9" 
     