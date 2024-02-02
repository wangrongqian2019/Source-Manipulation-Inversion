
### train(add training data and validation data here，format=h5py,size=(sample_size_train,,parameters.timespan, parameters.trace=3))
### parameters.trace is related to the CNN, must be 3 without modifying the neural network
data_path =
test_data_path =

result_path =

sample_size_train = 8000

learning_rate = 3e-4
end_lr = 1e-6

checkpoint_epoch = 0
num_epochs = 300
batchsize = 16
regular = 0

timespan = 2048
timespan_input = 4095
trace = 3

### test
receiver = 921
testnum = 307

val_data_path = test_data_path
Output_path='./OUTPUTS/'

test_checkpoint_epoch = 48

sample_id_test = 5400
sample_size_test = 1

num_epochs_test = 500
test_tag = 0
sample_size_val = 100
sample_id = 5000