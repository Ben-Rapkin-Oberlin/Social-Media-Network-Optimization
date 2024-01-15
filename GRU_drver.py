import tensorflow as tf
import numpy as np
import random
import GRU as gru
from RL_Classes import Environment



epochs=10
N=15
K=9
seed=1
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
num_nodes = 5  
time_steps = 8
env = Environment(num_nodes, N, K,seed)


#For now very basic loss of MSE of prior fitness and new fitness
model = gru.create_conv_lstm_model(num_nodes, time_steps,seed)
#model.compile(optimizer='adam', metrics=['accuracy'])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
data=env.prime(time_steps)
data = np.expand_dims(data, axis=0) 
old_score=data[0,time_steps-1,num_nodes,0]
mask=np.zeros((num_nodes))


for i in range(epochs):
    with tf.GradientTape() as tape:
        suggested = model(data)
    env.adj = tf.linalg.band_part(suggested[0, 0:-1, :, 0], 0, -1)  # upper tri
    env.adj = tf.linalg.set_diag(env.adj, mask)  # set diag to 0
    env.adj = env.adj + tf.transpose(env.adj)  # make symmetric
    
    score = float(env.step())
    loss = (num_nodes - score)**3
    loss = tf.cast(loss, dtype=tf.float32)
        #print('loss: ', loss, ' score: ', score)
    # Calculate gradients
    gradients = tape.gradient(loss, model.trainable_variables)

    # Apply gradients
    try:
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    except Exception as e:
        print(e)
        print('here')
    # Update data for the next iteration
    temp_score_tensor = tf.fill([1, num_nodes], score)
    temp_new_sample = tf.concat([env.adj, temp_score_tensor], axis=0)
    temp_new_sample = tf.reshape(temp_new_sample, [1, 1, num_nodes + 1, num_nodes])
    data = tf.concat([data[:, 1:, :, :], temp_new_sample], axis=1)  # drop oldest time step
    print('here')