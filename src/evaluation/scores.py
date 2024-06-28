from src.evaluation.eval_helper import *
from sklearn.metrics import roc_auc_score


def get_discriminative_score(real_train_dl, real_test_dl, fake_train_dl, fake_test_dl, config):
    mconfig = config.Evaluation.TestMetrics.discriminative_score

    class Discriminator(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, out_size=2):
            super(Discriminator, self).__init__()
            self.rnn = nn.GRU(input_size=input_size, num_layers=num_layers,
                              hidden_size=hidden_size, batch_first=True)
            self.linear = nn.Linear(hidden_size, out_size)

        def forward(self, x):
            x = self.rnn(x)[0][:, -1]
            return self.linear(x)

    train_dl = create_dl(real_train_dl, fake_train_dl, mconfig.batch_size, cutoff=False)
    test_dl = create_dl(real_test_dl, fake_test_dl, mconfig.batch_size, cutoff=False)

    pm = TrainValidateTestModel(epochs=mconfig.epochs, device=config.device)
    test_acc_list = []
    for i in range(1):
        model = Discriminator(train_dl.dataset[0][0].shape[-1], mconfig.hidden_size, mconfig.num_layers)
        _, _, test_acc = pm.train_val_test_classification(train_dl, test_dl, model, train=True, validate=True)
        test_acc_list.append(test_acc)
    mean_acc = np.mean(np.array(test_acc_list))
    std_acc = np.std(np.array(test_acc_list))
    return abs(mean_acc - 0.5), std_acc


def get_predictive_score(real_train_dl, real_test_dl, fake_train_dl, fake_test_dl, config):
    mconfig = config.Evaluation.TestMetrics.predictive_score

    train_dl = create_dl(fake_train_dl, fake_test_dl, mconfig.batch_size, cutoff=True)
    test_dl = create_dl(real_train_dl, real_test_dl, mconfig.batch_size, cutoff=True)
    
    class Predictor(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, out_size):
            super(Predictor, self).__init__()
            self.rnn = nn.LSTM(input_size=input_size, num_layers=num_layers,
                               hidden_size=hidden_size, batch_first=True)
            self.linear = nn.Linear(hidden_size, out_size)

        def forward(self, x):
            x = self.rnn(x)[0][:, -1]
            return self.linear(x)

    pm = TrainValidateTestModel(epochs=mconfig.epochs, device=config.device)
    test_loss_list = []
    for i in range(1):  ## Question: WHY 1?
        model = Predictor(train_dl.dataset[0][0].shape[-1],
                          mconfig.hidden_size,
                          mconfig.num_layers,
                          out_size=train_dl.dataset[0][1].shape[-1]
                          )
        model, test_loss = pm.train_val_test_regressor(
            train_dl=train_dl,
            test_dl=test_dl,
            model=model,
            train=True,
            validate=True
        )
        test_loss_list.append(test_loss)
    mean_loss = np.mean(np.array(test_loss_list))
    std_loss = np.std(np.array(test_loss_list))
    return mean_loss, std_loss


def get_classification_score(real_train_dl, real_test_dl, fake_train_dl, fake_test_dl, config):
    mconfig = config.Evaluation.TestMetrics.discriminative_score

    class Discriminator(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, out_size=2):
            super(Discriminator, self).__init__()
            self.rnn = nn.GRU(input_size=input_size, num_layers=num_layers,
                              hidden_size=hidden_size, batch_first=True)
            self.linear = nn.Linear(hidden_size, out_size)

        def forward(self, x):
            x = self.rnn(x)[0][:, -1]
            return self.linear(x)

    train_dl = create_dl(real_train_dl, fake_train_dl, mconfig.batch_size, cutoff=False)
    test_dl = create_dl(real_test_dl, fake_test_dl, mconfig.batch_size, cutoff=False)

    pm = TrainValidateTestModel(epochs=mconfig.epochs, device=config.device)
    test_acc_list = []
    for i in range(1):
        model = Discriminator(train_dl.dataset[0][0].shape[-1], mconfig.hidden_size, mconfig.num_layers)
        _, test_labels, test_acc = pm.train_val_test_classification(train_dl, test_dl, model, train=True, validate=True)
        test_acc_list.append(test_acc)
    return test_labels


def compute_auc(truth_crisis, fake_crisis, fake_regular, truth_regular, config, upsampling=True):

    train_set_size = int(0.8*truth_regular.shape[0])
    test_set_size = truth_crisis.shape[0] // 2

    if upsampling:
        crisis_training_set = torch.cat([truth_crisis[:test_set_size], fake_crisis])[:train_set_size]
    else:
        crisis_training_set = torch.cat([truth_crisis[:test_set_size]])[:train_set_size]
    regular_training_set = truth_regular[:train_set_size]

    crisis_training_dl = DataLoader(TensorDataset(crisis_training_set), batch_size=32, shuffle=True)
    regular_training_dl = DataLoader(TensorDataset(regular_training_set), batch_size=32, shuffle=True)

    crisis_test_dl = DataLoader(TensorDataset(truth_crisis[test_set_size:]), batch_size=4, shuffle=True)
    regular_test_dl = DataLoader(TensorDataset(truth_regular[train_set_size:train_set_size+test_set_size]), batch_size=4, shuffle=True)

    true_labels, pred_labels = get_classification_score(crisis_training_dl, crisis_test_dl, regular_training_dl, regular_test_dl, config)

    # print(true_labels, pred_labels)
    auc = roc_auc_score(true_labels.cpu().numpy(), pred_labels.cpu().numpy())

    return auc




# """Time-series Generative Adversarial Networks (TimeGAN) Codebase.

# Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
# "Time-series Generative Adversarial Networks," 
# Neural Information Processing Systems (NeurIPS), 2019.

# Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

# Last updated Date: April 24th 2020
# Code author: Jinsung Yoon (jsyoon0823@gmail.com)

# -----------------------------

# predictive_metrics.py

# Note: Use post-hoc RNN to classify original data and synthetic data

# Output: discriminative score (np.abs(classification accuracy - 0.5))
# """

# # Necessary Packages
# import tensorflow as tf
# import numpy as np
# from sklearn.metrics import accuracy_score
# from utils import train_test_divide, extract_time, batch_generator


# def train_test_divide (data_x, data_x_hat, data_t, data_t_hat, train_rate = 0.8):
#     """Divide train and test data for both original and synthetic data.

#     Args:
#     - data_x: original data
#     - data_x_hat: generated data
#     - data_t: original time
#     - data_t_hat: generated time
#     - train_rate: ratio of training data from the original data
#     """
#     # Divide train/test index (original data)
#     no = len(data_x)
#     idx = np.random.permutation(no)
#     train_idx = idx[:int(no*train_rate)]
#     test_idx = idx[int(no*train_rate):]

#     train_x = [data_x[i] for i in train_idx]
#     test_x = [data_x[i] for i in test_idx]
#     train_t = [data_t[i] for i in train_idx]
#     test_t = [data_t[i] for i in test_idx]      

#     # Divide train/test index (synthetic data)
#     no = len(data_x_hat)
#     idx = np.random.permutation(no)
#     train_idx = idx[:int(no*train_rate)]
#     test_idx = idx[int(no*train_rate):]

#     train_x_hat = [data_x_hat[i] for i in train_idx]
#     test_x_hat = [data_x_hat[i] for i in test_idx]
#     train_t_hat = [data_t_hat[i] for i in train_idx]
#     test_t_hat = [data_t_hat[i] for i in test_idx]

#     return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


# def extract_time (data):
#     """Returns Maximum sequence length and each sequence length.

#     Args:
#     - data: original data

#     Returns:
#     - time: extracted time information
#     - max_seq_len: maximum sequence length
#     """
#     time = list()
#     max_seq_len = 0
#     for i in range(len(data)):
#         max_seq_len = max(max_seq_len, len(data[i][:,0]))
#     time.append(len(data[i][:,0]))

#     return time, max_seq_len


# def batch_generator(data, time, batch_size):
#     """Mini-batch generator.

#     Args:
#     - data: time-series data
#     - time: time information
#     - batch_size: the number of samples in each batch

#     Returns:
#     - X_mb: time-series data in each batch
#     - T_mb: time information in each batch
#     """
#     no = len(data)
#     idx = np.random.permutation(no)
#     train_idx = idx[:batch_size]     
            
#     X_mb = list(data[i] for i in train_idx)
#     T_mb = list(time[i] for i in train_idx)

#     return X_mb, T_mb


# def discriminative_score_metrics (ori_data, generated_data):
#     """Use post-hoc RNN to classify original data and synthetic data

#     Args:
#     - ori_data: original data
#     - generated_data: generated synthetic data

#     Returns:
#     - discriminative_score: np.abs(classification accuracy - 0.5)
#     """
#     # Initialization on the Graph
#     tf.reset_default_graph()

#     # Basic Parameters
#     no, seq_len, dim = np.asarray(ori_data).shape    

#     # Set maximum sequence length and each sequence length
#     ori_time, ori_max_seq_len = extract_time(ori_data)
#     generated_time, generated_max_seq_len = extract_time(ori_data)
#     max_seq_len = max([ori_max_seq_len, generated_max_seq_len])  
        
#     ## Builde a post-hoc RNN discriminator network 
#     # Network parameters
#     hidden_dim = int(dim/2)
#     iterations = 2000
#     batch_size = 128

#     # Input place holders
#     # Feature
#     X = tf.placeholder(tf.float32, [None, max_seq_len, dim], name = "myinput_x")
#     X_hat = tf.placeholder(tf.float32, [None, max_seq_len, dim], name = "myinput_x_hat")

#     T = tf.placeholder(tf.int32, [None], name = "myinput_t")
#     T_hat = tf.placeholder(tf.int32, [None], name = "myinput_t_hat")

  

#     # discriminator function
#     def discriminator (x, t):
#         """Simple discriminator function.

#         Args:
#             - x: time-series data
#             - t: time information
            
#         Returns:
#             - y_hat_logit: logits of the discriminator output
#             - y_hat: discriminator output
#             - d_vars: discriminator variables
#         """
#         with tf.variable_scope("discriminator", reuse = tf.AUTO_REUSE) as vs:
#             d_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh, name = 'd_cell')
#             d_outputs, d_last_states = tf.nn.dynamic_rnn(d_cell, x, dtype=tf.float32, sequence_length = t)
#             y_hat_logit = tf.contrib.layers.fully_connected(d_last_states, 1, activation_fn=None) 
#             y_hat = tf.nn.sigmoid(y_hat_logit)
#             d_vars = [v for v in tf.all_variables() if v.name.startswith(vs.name)]

#         return y_hat_logit, y_hat, d_vars

#     y_logit_real, y_pred_real, d_vars = discriminator(X, T)
#     y_logit_fake, y_pred_fake, _ = discriminator(X_hat, T_hat)
        
#     # Loss for the discriminator
#     d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = y_logit_real, 
#                                                                         labels = tf.ones_like(y_logit_real)))
#     d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = y_logit_fake, 
#                                                                         labels = tf.zeros_like(y_logit_fake)))
#     d_loss = d_loss_real + d_loss_fake

#     # optimizer
#     d_solver = tf.train.AdamOptimizer().minimize(d_loss, var_list = d_vars)
        
#     ## Train the discriminator   
#     # Start session and initialize
#     sess = tf.Session()
#     sess.run(tf.global_variables_initializer())

#     # Train/test division for both original and generated data
#     train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat = \
#     train_test_divide(ori_data, generated_data, ori_time, generated_time)

#     # Training step
#     for itt in range(iterations):
            
#         # Batch setting
#         X_mb, T_mb = batch_generator(train_x, train_t, batch_size)
#         X_hat_mb, T_hat_mb = batch_generator(train_x_hat, train_t_hat, batch_size)
                
#         # Train discriminator
#         _, step_d_loss = sess.run([d_solver, d_loss], 
#                                     feed_dict={X: X_mb, T: T_mb, X_hat: X_hat_mb, T_hat: T_hat_mb})            

#     ## Test the performance on the testing set    
#     y_pred_real_curr, y_pred_fake_curr = sess.run([y_pred_real, y_pred_fake], 
#                                                 feed_dict={X: test_x, T: test_t, X_hat: test_x_hat, T_hat: test_t_hat})

#     y_pred_final = np.squeeze(np.concatenate((y_pred_real_curr, y_pred_fake_curr), axis = 0))
#     y_label_final = np.concatenate((np.ones([len(y_pred_real_curr),]), np.zeros([len(y_pred_fake_curr),])), axis = 0)

#     # Compute the accuracy
#     acc = accuracy_score(y_label_final, (y_pred_final>0.5))
#     discriminative_score = np.abs(0.5-acc)

#     return discriminative_score  


# """Time-series Generative Adversarial Networks (TimeGAN) Codebase.

# Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar, 
# "Time-series Generative Adversarial Networks," 
# Neural Information Processing Systems (NeurIPS), 2019.

# Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

# Last updated Date: April 24th 2020
# Code author: Jinsung Yoon (jsyoon0823@gmail.com)

# -----------------------------

# predictive_metrics.py

# Note: Use Post-hoc RNN to predict one-step ahead (last feature)
# """


# from sklearn.metrics import mean_absolute_error

# def predictive_score_metrics (ori_data, generated_data):
#     """Report the performance of Post-hoc RNN one-step ahead prediction.

#     Args:
#     - ori_data: original data
#     - generated_data: generated synthetic data

#     Returns:
#     - predictive_score: MAE of the predictions on the original data
#     """
#     # Initialization on the Graph
#     tf.reset_default_graph()

#     # Basic Parameters
#     no, seq_len, dim = np.asarray(ori_data).shape

#     # Set maximum sequence length and each sequence length
#     ori_time, ori_max_seq_len = extract_time(ori_data)
#     generated_time, generated_max_seq_len = extract_time(ori_data)
#     max_seq_len = max([ori_max_seq_len, generated_max_seq_len])  
        
#     ## Builde a post-hoc RNN predictive network 
#     # Network parameters
#     hidden_dim = int(dim/2)
#     iterations = 5000
#     batch_size = 128

#     # Input place holders
#     X = tf.placeholder(tf.float32, [None, max_seq_len-1, dim-1], name = "myinput_x")
#     T = tf.placeholder(tf.int32, [None], name = "myinput_t")    
#     Y = tf.placeholder(tf.float32, [None, max_seq_len-1, 1], name = "myinput_y")

#     # Predictor function
#     def predictor (x, t):
#         """Simple predictor function.

#         Args:
#             - x: time-series data
#             - t: time information
            
#         Returns:
#             - y_hat: prediction
#             - p_vars: predictor variables
#         """
#         with tf.variable_scope("predictor", reuse = tf.AUTO_REUSE) as vs:
#             p_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_dim, activation=tf.nn.tanh, name = 'p_cell')
#             p_outputs, p_last_states = tf.nn.dynamic_rnn(p_cell, x, dtype=tf.float32, sequence_length = t)
#             y_hat_logit = tf.contrib.layers.fully_connected(p_outputs, 1, activation_fn=None) 
#             y_hat = tf.nn.sigmoid(y_hat_logit)
#             p_vars = [v for v in tf.all_variables() if v.name.startswith(vs.name)]

#         return y_hat, p_vars

#     y_pred, p_vars = predictor(X, T)
#     # Loss for the predictor
#     p_loss = tf.losses.absolute_difference(Y, y_pred)
#     # optimizer
#     p_solver = tf.train.AdamOptimizer().minimize(p_loss, var_list = p_vars)
        
#     ## Training    
#     # Session start
#     sess = tf.Session()
#     sess.run(tf.global_variables_initializer())

#     # Training using Synthetic dataset
#     for itt in range(iterations):
            
#     # Set mini-batch
#     idx = np.random.permutation(len(generated_data))
#     train_idx = idx[:batch_size]     
            
#     X_mb = list(generated_data[i][:-1,:(dim-1)] for i in train_idx)
#     T_mb = list(generated_time[i]-1 for i in train_idx)
#     Y_mb = list(np.reshape(generated_data[i][1:,(dim-1)],[len(generated_data[i][1:,(dim-1)]),1]) for i in train_idx)        
            
#     # Train predictor
#     _, step_p_loss = sess.run([p_solver, p_loss], feed_dict={X: X_mb, T: T_mb, Y: Y_mb})        

#     ## Test the trained model on the original data
#     idx = np.random.permutation(len(ori_data))
#     train_idx = idx[:no]

#     X_mb = list(ori_data[i][:-1,:(dim-1)] for i in train_idx)
#     T_mb = list(ori_time[i]-1 for i in train_idx)
#     Y_mb = list(np.reshape(ori_data[i][1:,(dim-1)], [len(ori_data[i][1:,(dim-1)]),1]) for i in train_idx)

#     # Prediction
#     pred_Y_curr = sess.run(y_pred, feed_dict={X: X_mb, T: T_mb})

#     # Compute the performance in terms of MAE
#     MAE_temp = 0
#     for i in range(no):
#     MAE_temp = MAE_temp + mean_absolute_error(Y_mb[i], pred_Y_curr[i,:,:])

#     predictive_score = MAE_temp / no

#     return predictive_score
    