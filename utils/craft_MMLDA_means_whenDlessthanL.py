import tensorflow as tf

L = 100 #Number of classes
d = 512 #Dimension of features
lr = 0.0001 #Learning rate
mean_var = 10
steps = 10000 #optimization steps

z = tf.get_variable("auxiliary_variable", [d, L]) #dxL
x = z / tf.norm(z, axis=0, keepdims=True) #dxL, normalized in each column
XTX = tf.matmul(x, x, transpose_a=True) - 2 * tf.eye(L)#LxL, each element is the dot-product of two means, the diag elements are -1
cost = tf.reduce_max(XTX) #single element
opt = tf.train.AdamOptimizer(learning_rate=lr)
opt_op = opt.minimize(cost)
with tf.Session() as sess:
	sess.run(tf.initializers.global_variables())
	for i in range(steps):
		_, loss = sess.run([opt_op, cost])
		min_distance2 = loss
		print('Step %d, min_distance2: %f'%(i, min_distance2))

	mean_logits = sess.run(x)

mean_logits = mean_var * mean_logits.T 
import scipy.io as sio
sio.savemat('/mfs/tianyu/google/tianyu/kernelpara/dimmeans'+str(d)+'_meanvar'+str(mean_var)+'_class'+str(L)+'.mat', {'mean_logits': mean_logits})
#CUDA_VISIBLE_DEVICES=0 python craft_M3LDA_means.py	