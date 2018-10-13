import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
from plotting import newfig, savefig
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

class DeepVIV:
    # Initialize the class
    def __init__(self, t, x, y,
                       c, eta,
                       t_boundary, x_boundary, y_boundary, u_boundary, v_boundary,
                       layers_uvpc, layers_eta,
                       Re, Pec):
        
        self.Re = Re
        self.Pec = Pec
        
        X = np.concatenate([t, x, y], 1)
        self.X_min = X.min(0)
        self.X_max = X.max(0)
        
        # data on velocity (inside the domain)
        self.t = t
        self.x = x
        self.y = y
        self.c = c
        self.eta = eta
        
        # data on velocity (inlet)
        self.t_boundary = t_boundary
        self.x_boundary = x_boundary
        self.y_boundary = y_boundary
        self.u_boundary = u_boundary
        self.v_boundary = v_boundary
        
        # layers
        self.layers_uvpc = layers_uvpc
        self.layers_eta  = layers_eta
        
        # initialize NN
        self.weights_uvpc, self.biases_uvpc = self.initialize_NN(layers_uvpc)
        self.weights_eta, self.biases_eta = self.initialize_NN(layers_eta)
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        
        # placeholders for data on concentration (inside the domain)
        self.learning_rate = tf.placeholder(tf.float32, shape=[])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.y_tf = tf.placeholder(tf.float32, shape=[None, 1])        
        self.c_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.eta_tf = tf.placeholder(tf.float32, shape=[None, 1])
        
        self.dummy_tf = tf.placeholder(tf.float32, shape=(None, layers_uvpc[-1])) # dummy variable for fwd_gradients
        
        # placeholders for data on velocity (boundary)
        self.t_boundary_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.x_boundary_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.y_boundary_tf = tf.placeholder(tf.float32, shape=[None, 1])        
        self.u_boundary_tf = tf.placeholder(tf.float32, shape=[None, 1])
        self.v_boundary_tf = tf.placeholder(tf.float32, shape=[None, 1])
        
        # physics informed neural networks (inside the domain)
        (self.u_pred,
         self.v_pred,
         self.p_pred,
         self.c_pred,
         self.eta_pred,
         self.eq1_pred,
         self.eq2_pred,
         self.eq3_pred,
         self.eq4_pred) = self.net_VIV(self.t_tf, self.x_tf, self.y_tf)
        
        # physics informed neural networks (inlet)
        (self.u_boundary_pred,
         self.v_boundary_pred,
         _,
         _,
         _,
         _,
         _,
         _,
         _) = self.net_VIV(self.t_boundary_tf, self.x_boundary_tf, self.y_boundary_tf)
        
        # loss
        self.loss = tf.reduce_mean(tf.square(self.c_tf - self.c_pred)) + \
                    tf.reduce_mean(tf.square(self.eta_tf - self.eta_pred)) + \
                    tf.reduce_mean(tf.square(self.u_boundary_tf - self.u_boundary_pred)) + \
                    tf.reduce_mean(tf.square(self.v_boundary_tf - self.v_boundary_pred)) + \
                    tf.reduce_mean(tf.square(self.eq1_pred)) + \
                    tf.reduce_mean(tf.square(self.eq2_pred)) + \
                    tf.reduce_mean(tf.square(self.eq3_pred)) + \
                    tf.reduce_mean(tf.square(self.eq4_pred))
        
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
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = X
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.sin(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y
    
    def fwd_gradients(self, U, x):
        g = tf.gradients(U, x, grad_ys=self.dummy_tf)[0]
        return tf.gradients(g, self.dummy_tf)[0]
            
    def net_VIV(self, t, x, y):
        X = 2.0*(tf.concat([t,x,y], 1) - self.X_min)/(self.X_max - self.X_min) - 1
        uvpc = self.neural_net(X, self.weights_uvpc, self.biases_uvpc)
        
        t_tmp = 2.0*(t - self.X_min[0])/(self.X_max[0] - self.X_min[0]) - 1
        eta = self.neural_net(t_tmp, self.weights_eta, self.biases_eta)
        
        uvpc_t = self.fwd_gradients(uvpc, t)
        uvpc_x = self.fwd_gradients(uvpc, x)
        uvpc_y = self.fwd_gradients(uvpc, y)
        uvpc_xx = self.fwd_gradients(uvpc_x, x)
        uvpc_yy = self.fwd_gradients(uvpc_y, y)
        
        eta_t = tf.gradients(eta, t)[0]
        eta_tt = tf.gradients(eta_t, t)[0]
        
        u = uvpc[:,0:1]
        v = uvpc[:,1:2]
        p = uvpc[:,2:3]
        c = uvpc[:,3:4]
        
        u_t = uvpc_t[:,0:1]
        v_t = uvpc_t[:,1:2]
        c_t = uvpc_t[:,3:4]
        
        u_x = uvpc_x[:,0:1]
        v_x = uvpc_x[:,1:2]
        p_x = uvpc_x[:,2:3]
        c_x = uvpc_x[:,3:4]
        
        u_y = uvpc_y[:,0:1]
        v_y = uvpc_y[:,1:2]
        p_y = uvpc_y[:,2:3]
        c_y = uvpc_y[:,3:4]
        
        u_xx = uvpc_xx[:,0:1]
        v_xx = uvpc_xx[:,1:2]
        c_xx = uvpc_xx[:,3:4]
        
        u_yy = uvpc_yy[:,0:1]
        v_yy = uvpc_yy[:,1:2]
        c_yy = uvpc_yy[:,3:4]
        
        eq1 = c_t + (u*c_x + v*c_y) - (1.0/self.Pec)*(c_xx + c_yy) 
        eq2 = u_t + (u*u_x + v*u_y) + p_x - (1.0/self.Re)*(u_xx + u_yy) 
        eq3 = v_t + (u*v_x + v*v_y) + p_y - (1.0/self.Re)*(v_xx + v_yy) + eta_tt
        eq4 = u_x + v_y
        
        return u, v, p, c, eta, eq1, eq2, eq3, eq4
    
    def train(self, num_epochs, batch_size, learning_rate):

        for epoch in range(num_epochs):
            
            N = self.t.shape[0]
            perm = np.random.permutation(N)
            
            start_time = time.time()
            for it in range(0, N, batch_size):
                idx = perm[np.arange(it,it+batch_size)]
                (t_batch,
                 x_batch,
                 y_batch,
                 c_batch,
                 eta_batch) = (self.t[idx,:],
                               self.x[idx,:],
                               self.y[idx,:],
                               self.c[idx,:],
                               self.eta[idx,:])
                
                idx_boundary = np.random.choice(t_boundary.shape[0], batch_size)
                
                (t_boundary_batch,
                 x_boundary_batch,
                 y_boundary_batch,
                 u_boundary_batch,
                 v_boundary_batch) = (self.t_boundary[idx_boundary,:],
                                      self.x_boundary[idx_boundary,:],
                                      self.y_boundary[idx_boundary,:],
                                      self.u_boundary[idx_boundary,:],
                                      self.v_boundary[idx_boundary,:])

                tf_dict = {self.t_tf: t_batch, self.x_tf: x_batch, self.y_tf: y_batch,
                           self.c_tf: c_batch, self.eta_tf: eta_batch,
                           self.t_boundary_tf: t_boundary_batch, self.x_boundary_tf: x_boundary_batch, self.y_boundary_tf: y_boundary_batch,
                           self.u_boundary_tf: u_boundary_batch, self.v_boundary_tf: v_boundary_batch,
                           self.dummy_tf: np.ones((batch_size, self.layers_uvpc[-1])),
                           self.learning_rate: learning_rate}
                
                self.sess.run(self.train_op, tf_dict)
                
                # Print
                if it % (10*batch_size) == 0:
                    elapsed = time.time() - start_time
                    loss_value, learning_rate_value = self.sess.run([self.loss,self.learning_rate], tf_dict)
                    print('Epoch: %d, It: %d, Loss: %.3e, Time: %.2f, Learning Rate: %.3e'
                          %(epoch, it/batch_size, loss_value, elapsed, learning_rate_value))
                    start_time = time.time()
    
    def predict(self, t_star, x_star, y_star):
        
        tf_dict = {self.t_tf: t_star, self.x_tf: x_star, self.y_tf: y_star}
        
        u_star = self.sess.run(self.u_pred, tf_dict)
        v_star = self.sess.run(self.v_pred, tf_dict)
        p_star = self.sess.run(self.p_pred, tf_dict)
        c_star = self.sess.run(self.c_pred, tf_dict)
        eta_star = self.sess.run(self.eta_pred, tf_dict)
        
        return u_star, v_star, p_star, c_star, eta_star
    
    def predict_drag_lift(self, t_cyl):
        
        viscosity = (1.0/self.Re)
        
        theta = np.linspace(0.0,2*np.pi,200)[:,None] # N x 1
        d_theta = theta[1,0] - theta[0,0]
        x_cyl = 0.5*np.cos(theta) # N x 1
        y_cyl = 0.5*np.sin(theta) # N x 1
            
        N = x_cyl.shape[0]
        T = t_cyl.shape[0]
        
        T_star = np.tile(t_cyl, (1,N)).T # N x T
        X_star = np.tile(x_cyl, (1,T)) # N x T
        Y_star = np.tile(y_cyl, (1,T)) # N x T
        
        t_star = np.reshape(T_star,[-1,1]) # NT x 1
        x_star = np.reshape(X_star,[-1,1]) # NT x 1
        y_star = np.reshape(Y_star,[-1,1]) # NT x 1
        
        u_x_pred = tf.gradients(self.u_pred, self.x_tf)[0]
        u_y_pred = tf.gradients(self.u_pred, self.y_tf)[0]
        
        v_x_pred = tf.gradients(self.v_pred, self.x_tf)[0]
        v_y_pred = tf.gradients(self.v_pred, self.y_tf)[0]
        
        tf_dict = {self.t_tf: t_star, self.x_tf: x_star, self.y_tf: y_star}
        
        p_star, u_x_star, u_y_star, v_x_star, v_y_star = self.sess.run([self.p_pred, u_x_pred, u_y_pred, v_x_pred, v_y_pred], tf_dict)
        
        P_star = np.reshape(p_star, [N,T]) # N x T
        P_star = P_star - np.mean(P_star, axis=0)
        U_x_star = np.reshape(u_x_star, [N,T]) # N x T
        U_y_star = np.reshape(u_y_star, [N,T]) # N x T
        V_x_star = np.reshape(v_x_star, [N,T]) # N x T
        V_y_star = np.reshape(v_y_star, [N,T]) # N x T
    
        INT0 = (-P_star[0:-1,:] + 2*viscosity*U_x_star[0:-1,:])*X_star[0:-1,:] + viscosity*(U_y_star[0:-1,:] + V_x_star[0:-1,:])*Y_star[0:-1,:]
        INT1 = (-P_star[1: , :] + 2*viscosity*U_x_star[1: , :])*X_star[1: , :] + viscosity*(U_y_star[1: , :] + V_x_star[1: , :])*Y_star[1: , :]
            
        F_D = 0.5*np.sum(INT0.T+INT1.T, axis = 1)*d_theta # T x 1
    
        
        INT0 = (-P_star[0:-1,:] + 2*viscosity*V_y_star[0:-1,:])*Y_star[0:-1,:] + viscosity*(U_y_star[0:-1,:] + V_x_star[0:-1,:])*X_star[0:-1,:]
        INT1 = (-P_star[1: , :] + 2*viscosity*V_y_star[1: , :])*Y_star[1: , :] + viscosity*(U_y_star[1: , :] + V_x_star[1: , :])*X_star[1: , :]
            
        F_L = 0.5*np.sum(INT0.T+INT1.T, axis = 1)*d_theta # T x 1
            
        return F_D, F_L
    
def plot_solution(x_star, y_star, u_star, ax):
    
    nn = 200
    x = np.linspace(x_star.min(), x_star.max(), nn)
    y = np.linspace(y_star.min(), y_star.max(), nn)
    X, Y = np.meshgrid(x,y)
    
    X_star = np.concatenate((x_star, y_star), axis=1)
    
    U_star = griddata(X_star, u_star.flatten(), (X, Y), method='linear')
    
    # h = ax.pcolor(X,Y,U_star, cmap = 'jet')
    
    h = ax.imshow(U_star, interpolation='nearest', cmap='jet', 
                  extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()],
                  origin='lower', aspect='auto')
    
    return h
    
if __name__ == "__main__": 
    
    N_train  = 4000000
        
    layers_uvpc = [3] + 10*[4*32] + [4]
    layers_eta  = [1] + 10*[1*32] + [1]
    
    # Load Data
    data = scipy.io.loadmat('./Data/VIV_Concentration.mat')
        
    t_star = data['t_star'] # T x 1
    eta_star = data['eta_star'] # T x 1
    
    T = t_star.shape[0]

    X_star = data['X_star']
    Y_star = data['Y_star']        
    U_star = data['U_star']
    V_star = data['V_star']
    P_star = data['P_star']
    C_star = data['C_star']
        
    t = np.concatenate([t_star[i]+0.0*X_star[i,0] for i in range(0,T)])
    x = np.concatenate([X_star[i,0] for i in range(0,T)])
    y = np.concatenate([Y_star[i,0] for i in range(0,T)])
    u = np.concatenate([U_star[i,0] for i in range(0,T)])
    v = np.concatenate([V_star[i,0] for i in range(0,T)])
    p = np.concatenate([P_star[i,0] for i in range(0,T)])
    c = np.concatenate([C_star[i,0] for i in range(0,T)])
    eta = np.concatenate([eta_star[i]+0.0*X_star[i,0] for i in range(0,T)])
    
    t_inlet = np.concatenate([(t_star[i]+0.0*X_star[i,0])[X_star[i,0] == X_star[i,0].min()][:,None] for i in range(0,T)])
    x_inlet = np.concatenate([X_star[i,0][X_star[i,0] == X_star[i,0].min()][:,None] for i in range(0,T)])
    y_inlet = np.concatenate([Y_star[i,0][X_star[i,0] == X_star[i,0].min()][:,None] for i in range(0,T)])
    u_inlet = np.concatenate([U_star[i,0][X_star[i,0] == X_star[i,0].min()][:,None] for i in range(0,T)])
    v_inlet = np.concatenate([V_star[i,0][X_star[i,0] == X_star[i,0].min()][:,None] for i in range(0,T)])
    p_inlet = np.concatenate([P_star[i,0][X_star[i,0] == X_star[i,0].min()][:,None] for i in range(0,T)])
    c_inlet = np.concatenate([C_star[i,0][X_star[i,0] == X_star[i,0].min()][:,None] for i in range(0,T)])
    eta_inlet = np.concatenate([(eta_star[i]+0.0*X_star[i,0])[X_star[i,0] == X_star[i,0].min()][:,None] for i in range(0,T)])

    t_outlet = np.concatenate([(t_star[i]+0.0*X_star[i,0])[X_star[i,0] == X_star[i,0].max()][:,None] for i in range(0,T)])
    x_outlet = np.concatenate([X_star[i,0][X_star[i,0] == X_star[i,0].max()][:,None] for i in range(0,T)])
    y_outlet = np.concatenate([Y_star[i,0][X_star[i,0] == X_star[i,0].max()][:,None] for i in range(0,T)])
    u_outlet = np.concatenate([U_star[i,0][X_star[i,0] == X_star[i,0].max()][:,None] for i in range(0,T)])
    v_outlet = np.concatenate([V_star[i,0][X_star[i,0] == X_star[i,0].max()][:,None] for i in range(0,T)])
    p_outlet = np.concatenate([P_star[i,0][X_star[i,0] == X_star[i,0].max()][:,None] for i in range(0,T)])
    c_outlet = np.concatenate([C_star[i,0][X_star[i,0] == X_star[i,0].max()][:,None] for i in range(0,T)])
    eta_outlet = np.concatenate([(eta_star[i]+0.0*X_star[i,0])[X_star[i,0] == X_star[i,0].max()][:,None] for i in range(0,T)])

    t_bottom = np.concatenate([(t_star[i]+0.0*X_star[i,0])[Y_star[i,0] == Y_star[i,0].min()][:,None] for i in range(0,T)])
    x_bottom = np.concatenate([X_star[i,0][Y_star[i,0] == Y_star[i,0].min()][:,None] for i in range(0,T)])
    y_bottom = np.concatenate([Y_star[i,0][Y_star[i,0] == Y_star[i,0].min()][:,None] for i in range(0,T)])
    u_bottom = np.concatenate([U_star[i,0][Y_star[i,0] == Y_star[i,0].min()][:,None] for i in range(0,T)])
    v_bottom = np.concatenate([V_star[i,0][Y_star[i,0] == Y_star[i,0].min()][:,None] for i in range(0,T)])
    p_bottom = np.concatenate([P_star[i,0][Y_star[i,0] == Y_star[i,0].min()][:,None] for i in range(0,T)])
    c_bottom = np.concatenate([C_star[i,0][Y_star[i,0] == Y_star[i,0].min()][:,None] for i in range(0,T)])
    eta_bottom = np.concatenate([(eta_star[i]+0.0*X_star[i,0])[Y_star[i,0] == Y_star[i,0].min()][:,None] for i in range(0,T)])

    t_top = np.concatenate([(t_star[i]+0.0*X_star[i,0])[Y_star[i,0] == Y_star[i,0].max()][:,None] for i in range(0,T)])
    x_top = np.concatenate([X_star[i,0][Y_star[i,0] == Y_star[i,0].max()][:,None] for i in range(0,T)])
    y_top = np.concatenate([Y_star[i,0][Y_star[i,0] == Y_star[i,0].max()][:,None] for i in range(0,T)])
    u_top = np.concatenate([U_star[i,0][Y_star[i,0] == Y_star[i,0].max()][:,None] for i in range(0,T)])
    v_top = np.concatenate([V_star[i,0][Y_star[i,0] == Y_star[i,0].max()][:,None] for i in range(0,T)])
    p_top = np.concatenate([P_star[i,0][Y_star[i,0] == Y_star[i,0].max()][:,None] for i in range(0,T)])
    c_top = np.concatenate([C_star[i,0][Y_star[i,0] == Y_star[i,0].max()][:,None] for i in range(0,T)])
    eta_top = np.concatenate([(eta_star[i]+0.0*X_star[i,0])[Y_star[i,0] == Y_star[i,0].max()][:,None] for i in range(0,T)])
    
    t_cyl = np.concatenate([(t_star[i]+0.0*X_star[i,0])[X_star[i,0]**2 + Y_star[i,0]**2 == 0.25][:,None] for i in range(0,T)])
    x_cyl = np.concatenate([X_star[i,0][X_star[i,0]**2 + Y_star[i,0]**2 == 0.25][:,None] for i in range(0,T)])
    y_cyl = np.concatenate([Y_star[i,0][X_star[i,0]**2 + Y_star[i,0]**2 == 0.25][:,None] for i in range(0,T)])
    u_cyl = np.concatenate([U_star[i,0][X_star[i,0]**2 + Y_star[i,0]**2 == 0.25][:,None] for i in range(0,T)])
    v_cyl = np.concatenate([V_star[i,0][X_star[i,0]**2 + Y_star[i,0]**2 == 0.25][:,None] for i in range(0,T)])
    p_cyl = np.concatenate([P_star[i,0][X_star[i,0]**2 + Y_star[i,0]**2 == 0.25][:,None] for i in range(0,T)])
    c_cyl = np.concatenate([C_star[i,0][X_star[i,0]**2 + Y_star[i,0]**2 == 0.25][:,None] for i in range(0,T)])
    eta_cyl = np.concatenate([(eta_star[i]+0.0*X_star[i,0])[X_star[i,0]**2 + Y_star[i,0]**2 == 0.25][:,None] for i in range(0,T)])
    
    ######################################################################
    ######################## Noiseles Data ###############################
    ######################################################################
    # Training Data
    idx = np.random.choice(t.shape[0], N_train, replace=False)
    t_train = t[idx,:]
    x_train = x[idx,:]
    y_train = y[idx,:]
    u_train = u[idx,:]
    v_train = v[idx,:]
    p_train = p[idx,:]
    c_train = c[idx,:]
    eta_train = eta[idx,:]
    
    # Training Data on velocity (boundary)
    
    t_boundary = np.concatenate((t_inlet,t_outlet,t_top,t_bottom,t_cyl))
    x_boundary = np.concatenate((x_inlet,x_outlet,x_top,x_bottom,x_cyl))
    y_boundary = np.concatenate((y_inlet,y_outlet,y_top,y_bottom,y_cyl))
    u_boundary = np.concatenate((u_inlet,u_outlet,u_top,u_bottom,u_cyl))
    v_boundary = np.concatenate((v_inlet,v_outlet,v_top,v_bottom,v_cyl))
        
    # Training
    model = DeepVIV(t_train, x_train, y_train,
                    c_train, eta_train,
                    t_boundary, x_boundary, y_boundary, u_boundary, v_boundary,
                    layers_uvpc, layers_eta,
                    Re = 100, Pec = 90)
        
    model.train(num_epochs = 200, batch_size = 10000, learning_rate=1e-3)
    model.train(num_epochs = 300, batch_size = 10000, learning_rate=1e-4)
    model.train(num_epochs = 300, batch_size = 10000, learning_rate=1e-5)
    model.train(num_epochs = 200, batch_size = 10000, learning_rate=1e-6)
    
    F_D, F_L = model.predict_drag_lift(t_star)
    
    fig, ax1 = plt.subplots()
    ax1.plot(t_star, F_D, 'b')
    ax1.set_xlabel('$t$')
    ax1.set_ylabel('$F_D$', color='b')
    ax1.tick_params('y', colors='b')
    
    ax2 = ax1.twinx()
    ax2.plot(t_star, F_L, 'r')
    ax2.set_ylabel('$F_L$', color='r')
    ax2.tick_params('y', colors='r')
    
    fig.tight_layout()
    
    # savefig('./Figures/VIV_data_on_concentration_lift_drag', crop = False)
    
    # Test Data
    snap = 100
    t_test = t_star[snap] + 0.0*X_star[snap,0]
    x_test = X_star[snap,0]
    y_test = Y_star[snap,0]
    
    u_test = U_star[snap,0]
    v_test = V_star[snap,0]
    p_test = P_star[snap,0]
    c_test = C_star[snap,0]
    eta_test = eta_star[snap] + 0.0*X_star[snap,0]
    
    # Prediction
    u_pred, v_pred, p_pred, c_pred, eta_pred = model.predict(t_test, x_test, y_test)
    
    # Error
    error_u = np.linalg.norm(u_test-u_pred,2)/np.linalg.norm(u_test,2)
    error_v = np.linalg.norm(v_test-v_pred,2)/np.linalg.norm(v_test,2)
    error_p = np.linalg.norm(p_test-p_pred,2)/np.linalg.norm(p_test,2)
    error_c = np.linalg.norm(c_test-c_pred,2)/np.linalg.norm(c_test,2)
    error_eta = np.linalg.norm(eta_test-eta_pred,2)/np.linalg.norm(eta_test,2)

    print('Error u: %e' % (error_u))
    print('Error v: %e' % (error_v))
    print('Error p: %e' % (error_p))
    print('Error c: %e' % (error_c))
    print('Error eta: %e' % (error_eta))
        
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################
    
    circle11 = plt.Circle((0, 0), 0.5, facecolor='w', edgecolor='k')
    circle12 = plt.Circle((0, 0), 0.5, facecolor='w', edgecolor='k')
    circle21 = plt.Circle((0, 0), 0.5, facecolor='w', edgecolor='k')
    circle22 = plt.Circle((0, 0), 0.5, facecolor='w', edgecolor='k')
    circle31 = plt.Circle((0, 0), 0.5, facecolor='w', edgecolor='k')
    circle32 = plt.Circle((0, 0), 0.5, facecolor='w', edgecolor='k')
    circle41 = plt.Circle((0, 0), 0.5, facecolor='w', edgecolor='k')
    circle42 = plt.Circle((0, 0), 0.5, facecolor='w', edgecolor='k')
    
    fig, ax = newfig(1.0, 1.6)
    ax.axis('off')

    gs = gridspec.GridSpec(4, 2)
    gs.update(top=0.95, bottom=0.07, left=0.1, right=0.9, wspace=0.5, hspace=0.7)
    
    ########      Exact c(t,x,y)     ###########     
    ax = plt.subplot(gs[0:1, 0])
    h = plot_solution(x_test,y_test,c_test,ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.add_artist(circle11)
    ax.axis('equal')
    
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Exact $c(t,x,y)$', fontsize = 10)
    
    ########     Learned c(t,x,y)     ###########
    ax = plt.subplot(gs[0:1, 1])
    h = plot_solution(x_test,y_test,c_pred,ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.add_artist(circle12)
    ax.axis('equal')
    
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Learned $c(t,x,y)$', fontsize = 10)
    
    ########      Exact u(t,x,y)     ###########     
    ax = plt.subplot(gs[1:2, 0])
    h = plot_solution(x_test,y_test,u_test,ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.add_artist(circle21)
    ax.axis('equal')
    
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Exact $u(t,x,y)$', fontsize = 10)
    
    ########     Learned u(t,x,y)     ###########
    ax = plt.subplot(gs[1:2, 1])
    h = plot_solution(x_test,y_test,u_pred,ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.add_artist(circle22)
    ax.axis('equal')
    
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Learned $u(t,x,y)$', fontsize = 10)
    
    ########      Exact v(t,x,y)     ###########     
    ax = plt.subplot(gs[2:3, 0])
    h = plot_solution(x_test,y_test,v_test,ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.add_artist(circle31)
    ax.axis('equal')
    
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Exact $v(t,x,y)$', fontsize = 10)
    
    ########     Learned v(t,x,y)     ###########
    ax = plt.subplot(gs[2:3, 1])
    h = plot_solution(x_test,y_test,v_pred,ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.add_artist(circle32)
    ax.axis('equal')
    
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Learned $v(t,x,y)$', fontsize = 10)
    
    ########      Exact p(t,x,y)     ###########     
    ax = plt.subplot(gs[3:4, 0])
    h = plot_solution(x_test,y_test,p_test,ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.add_artist(circle41)
    ax.axis('equal')    
    
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Exact $p(t,x,y)$', fontsize = 10)
    
    ########     Learned p(t,x,y)     ###########
    ax = plt.subplot(gs[3:4, 1])
    h = plot_solution(x_test,y_test,p_pred,ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.add_artist(circle42)
    ax.axis('equal')
        
    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Learned $p(t,x,y)$', fontsize = 10)
    
    # savefig('./Figures/VIV_data_on_concentration', crop = False)
    
    ################ Save Data ###########################
    
    U_pred = np.zeros((T,), dtype=np.object)
    V_pred = np.zeros((T,), dtype=np.object)
    P_pred = np.zeros((T,), dtype=np.object)
    C_pred = np.zeros((T,), dtype=np.object)
    Eta_pred = np.zeros((T,), dtype=np.object)
    for snap in range(0,t_star.shape[0]):
        t_test = t_star[snap] + 0.0*X_star[snap,0]
        x_test = X_star[snap,0]
        y_test = Y_star[snap,0]
        
        u_test = U_star[snap,0]
        v_test = V_star[snap,0]
        p_test = P_star[snap,0]
        c_test = C_star[snap,0]
        eta_test = eta_star[snap] + 0.0*X_star[snap,0]
    
        # Prediction
        u_pred, v_pred, p_pred, c_pred, eta_pred = model.predict(t_test, x_test, y_test)
        
        U_pred[snap] = u_pred
        V_pred[snap] = v_pred
        P_pred[snap] = p_pred
        C_pred[snap] = c_pred
        Eta_pred[snap] = eta_pred
    
        # Error
        error_u = np.linalg.norm(u_test-u_pred,2)/np.linalg.norm(u_test,2)
        error_v = np.linalg.norm(v_test-v_pred,2)/np.linalg.norm(v_test,2)
        error_p = np.linalg.norm(p_test-p_pred,2)/np.linalg.norm(p_test,2)
        error_c = np.linalg.norm(c_test-c_pred,2)/np.linalg.norm(c_test,2)
        error_eta = np.linalg.norm(eta_test-eta_pred,2)/np.linalg.norm(eta_test,2)
    
        print('Error u: %e' % (error_u))
        print('Error v: %e' % (error_v))
        print('Error p: %e' % (error_p))
        print('Error c: %e' % (error_c))
        print('Error eta: %e' % (error_eta))
    
    scipy.io.savemat('./Results/VIV_data_on_concentration_results_%s.mat' %(time.strftime('%d_%m_%Y')),
                     {'U_pred':U_pred, 'V_pred':V_pred, 'P_pred':P_pred, 'C_pred':C_pred, 'Eta_pred':Eta_pred, 'F_L':F_L, 'F_D':F_D})
