Simulation control probe with DQN

There are four code files:

Project2_PID

Project2_RL

Project2_RL+NN

Project2_NN_TEST

Project2_PID:

This program is to simulate the operation of PID control method, it has the same
probe model as other files. Just operate it , it will give a result image and a
gif of control process. The error value will also be shown.

Library: numpy , matplotlib

Project2_RL:

This program is to control the probe using traditional Q-learning method. We can
find some shortage of this method from its result.Just operate it, it will give
a result image and gif of control process. The error value will also be
shown.(The testing map is different from training map,the result is not good
,reason included in )

Library: numpy , matplotlib, random

Project2_RL+NN:

The training file of DQN, operate, it will load the weight files
(‘my_model_p20_15.h5’) and continue training, after each successful episode, the
model will be save in the save file. The training map is the same as above
two(PID RL ) algorithm.

Library: numpy , matplotlib, random, keras, collection

Project2_NN_TEST:

The test file of the DQN, has the same structure as the training file, and load
‘my_model_p20_15.h5’ file. While do not invoke update function and the epsilon
is zero. The test map is the same as RL to make comparison. Just operate it will
give the result image and gif of control process.

Library: numpy , matplotlib, random, keras, collections

The gif:

Result for PID play very slowly(It had been accelerated 10X),for it 1000 working
period .Other two gif is faster for its working period is 15.
