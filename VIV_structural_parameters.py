import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import time

class DeepVIV:
    # Initialize the class
    def __init__(self, t, eta, lift,
                 layers):
        
        self.X_min = t.min(0)
        self.X_max = t.max(0)
        
        # data
        self.t = t
        self.eta = eta
        self.lift = lift
        
        # layers
        self.layers = layers
        
        # initialize NN
        self.weights, self.biases = self.initialize_NN(layers)
        
        self.rho = 2.0 # tf.Variable(tf.ones([1], dtype=tf.float32), dtype=tf.float32)
        self.b = tf.Variable(0.05*tf.ones([1], dtype=tf.float32), dtype=tf.float32)
        self.k = tf.Variable(2.0*tf.ones([1], dtype=tf.float32), dtype=tf.float32)
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        # placeholders for data
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.eta_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.lift_tf = tf.placeholder(tf.float32, shape=[None, 1])
                
        # physics informed neural networks
        (self.eta_pred,
         self.lift_pred) = self.net_structure(self.t_tf)
        
        # loss
        self.loss = tf.reduce_sum(tf.square(self.eta_tf - self.eta_pred)) + \
                    tf.reduce_sum(tf.square(self.lift_tf - self.lift_pred))
        
        # optimizers
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev, dtype=tf.float32), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X-self.X_min)/(self.X_max-self.X_min) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.sin(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
            
    def net_structure(self, t):
        eta = self.neural_net(t, self.weights, self.biases)
                        
        eta_t = tf.gradients(eta, t)[0]
        eta_tt = tf.gradients(eta_t, t)[0]
        
        lift = self.rho*eta_tt + self.b*eta_t + self.k*eta
        
        return eta, lift
    
    def train(self, num_epochs, batch_size, learning_rate):

        for epoch in range(num_epochs):
            
            N = self.t.shape[0]
            perm = np.random.permutation(N)
            
            start_time = time.time()
            for it in range(0, N, batch_size):
                idx = perm[np.arange(it,it+batch_size)]
                (t_batch,
                 eta_batch,
                 lift_batch) = (self.t[idx,:],
                                self.eta[idx,:],
                                self.lift[idx,:])

                tf_dict = {self.t_tf: t_batch, self.eta_tf: eta_batch, self.lift_tf: lift_batch, self.learning_rate: learning_rate}
                
                self.sess.run(self.train_op, tf_dict)
                
                # Print
                if it % (10*batch_size) == 0:
                    elapsed = time.time() - start_time
                    loss_value, b_value, k_value, learning_rate_value = self.sess.run([self.loss, self.b, self.k, self.learning_rate], tf_dict)
                    print('Epoch: %d, It: %d, Loss: %.3e, b: %.5f, k: %.3f, Time: %.2f, Learning Rate: %.3e'
                          %(epoch, it/batch_size, loss_value, b_value, k_value, elapsed, learning_rate_value))
                    start_time = time.time()
    
    def predict(self, t_star):
        
        tf_dict = {self.t_tf: t_star}
        
        eta_star = self.sess.run(self.eta_pred, tf_dict)
        lift_star = self.sess.run(self.lift_pred, tf_dict)
        
        return eta_star, lift_star
    
if __name__ == "__main__": 
    
    layers  = [1] + 10*[32] + [1]
    
    # Load Exact Data
    data = scipy.io.loadmat('./Data/VIV_displacement_lift_drag.mat')
    t_star = data['t_structure'] # T x 1
    eta_star = data['eta_structure'] # T x 1
    lift_star = data['lift_structure'] # T x 1
    drag_star = data['drag_structure'] # T x 1

    # Load Approximate Data (velocities)
#    data = scipy.io.loadmat('./Data/VIV_Concentration.mat')
#    data_results = scipy.io.loadmat('./Results/VIV_data_on_velocities_results_10_06_2018.mat')
#    t_star = data['t_star'] # T x 1
#    eta_star = data['eta_star'] # T x 1
#    lift_star = data_results['F_L'].T # T x 1
#    drag_star = data_results['F_D'].T # T x 1

    # Load Approximate Data (concentration)
#    data = scipy.io.loadmat('./Data/VIV_Concentration.mat')
#    data_results = scipy.io.loadmat('./Results/VIV_data_on_concentration_results_10_06_2018.mat')
#    t_star = data['t_star'] # T x 1
#    eta_star = data['eta_star'] # T x 1
#    lift_star = data_results['F_L'].T # T x 1
#    drag_star = data_results['F_D'].T # T x 1
    
    N_train  = t_star.shape[0]
    
#    plt.figure()
#    plt.subplot(221)
#    plt.plot(t_star,eta_star)
#    plt.subplot(223)
#    plt.plot(t_star,lift_star)
#    plt.subplot(224)
#    plt.plot(t_star,drag_star)
    
    T = t_star.shape[0]
        
    t = t_star
    eta = eta_star
    lift = lift_star
    drag = drag_star
    
    ######################################################################
    ######################## Noiseles Data ###############################
    ######################################################################
    # Training Data
    idx = np.random.choice(t.shape[0], N_train, replace=False)
    t_train = t[idx,:]
    eta_train = eta[idx,:]
    lift_train = lift[idx,:]
    drag_train = drag[idx,:]
    
    # Training
    model = DeepVIV(t_train, eta_train, lift_train, layers)
        
    model.train(num_epochs = 20000, batch_size = N_train, learning_rate=1e-3)
    model.train(num_epochs = 30000, batch_size = N_train, learning_rate=1e-4)
    model.train(num_epochs = 30000, batch_size = N_train, learning_rate=1e-5)
    model.train(num_epochs = 20000, batch_size = N_train, learning_rate=1e-6)
    
    eta, lift = model.predict(t_star)
    
    fig, ax1 = plt.subplots()
    ax1.plot(t_star, eta, 'b')
    ax1.plot(t_star, eta_star, 'r--')
    ax1.set_xlabel('$t$')
    ax1.set_ylabel('$\eta$', color='b')
    ax1.tick_params('y', colors='b')
    
    ax2 = ax1.twinx()
    ax2.plot(t_star, lift, 'k')
    ax2.plot(t_star, lift_star, 'r--')
    ax2.set_ylabel('$F_L$', color='r')
    ax2.tick_params('y', colors='r')
    
    fig.tight_layout()
    
    k_exact = 2.202
    b_exact = 0.084
    
    k_pred = model.sess.run(model.k)
    b_pred = model.sess.run(model.b)
    
    k_error = np.abs(k_exact - k_pred)/np.abs(k_exact)
    b_error = np.abs(b_exact - b_pred)/np.abs(b_exact)
    
    scipy.io.savemat('./Results/VIV_structural_parameters_results_%s.mat' %(time.strftime('%d_%m_%Y')),
                     {'eta':eta, 'lift':lift})