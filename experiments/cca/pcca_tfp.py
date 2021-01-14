import functools
import warnings

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

from tensorflow_probability import bijectors as tfb
from tensorflow_probability import distributions as tfd

tf.enable_v2_behavior()

plt.style.use("ggplot")
warnings.filterwarnings('ignore')


Root = tfd.JointDistributionCoroutine.Root

def pcca_model(data_dim_x1, data_dim_x2, latent_dim, num_datapoints, stddv_datapoints):

  def pcca(data_dim_x1, data_dim_x2, latent_dim, num_datapoints, stddv_datapoints):

    # Exclusive LVs for dataset 1
    z1 = yield tfd.Independent(tfd.Normal(loc=tf.zeros([latent_dim, num_datapoints]),
                   scale=tf.ones([latent_dim, num_datapoints]),
                   name="z1"))

    # Exclusive LVs for dataset 2
    z2 = yield tfd.Independent(tfd.Normal(loc=tf.zeros([latent_dim, num_datapoints]),
                   scale=tf.ones([latent_dim, num_datapoints]),
                   name="z2"))

    # Shared LVs
    zshared = yield tfd.Independent(tfd.Normal(loc=tf.zeros([latent_dim, num_datapoints]),
                   scale=tf.ones([latent_dim, num_datapoints]),
                   name="zshared"))

    # Mapping from shared LVs to dataset 1
    lambda1 = yield tfd.Independent(tfd.Normal(loc=tf.zeros([data_dim_x1, latent_dim]),
                   scale=tf.ones([data_dim_x1, latent_dim]),
                   name="lambda1"))

    # Mapping from shared LVs to dataset 2
    lambda2 = yield tfd.Independent(tfd.Normal(loc=tf.zeros([data_dim_x2, latent_dim]),
                   scale=tf.ones([data_dim_x2, latent_dim]),
                   name="lambda2"))

    # Mapping from z1 to dataset 1
    b1 = yield tfd.Independent(tfd.Normal(loc=tf.zeros([data_dim_x1, latent_dim]),
                   scale=tf.ones([data_dim_x1, latent_dim]),
                   name="b1"))

    # Mapping from z2 to dataset 2
    b2 = yield tfd.Independent(tfd.Normal(loc=tf.zeros([data_dim_x2, latent_dim]),
                   scale=tf.ones([data_dim_x2, latent_dim]),
                   name="b2"))

    # Dataset 1
    x1 = yield tfd.Independent(tfd.Normal(loc=tf.matmul(lambda1, zshared) + tf.matmul(b1, z1),
                         scale=stddv_datapoints,
                         name="x1"))

    # Dataset 2
    x2 = yield tfd.Independent(tfd.Normal(loc=tf.matmul(lambda2, zshared) + tf.matmul(b2, z2),
                         scale=stddv_datapoints,
                         name="x2"))
    
  concrete_pcca_model = functools.partial(pcca,
      data_dim_x1=data_dim_x1,
      data_dim_x2=data_dim_x2,
      latent_dim=latent_dim,
      num_datapoints=num_datapoints,
      stddv_datapoints=stddv_datapoints)

  model = tfd.JointDistributionCoroutineAutoBatched(concrete_pcca_model)
  # model = tfd.JointDistributionCoroutine(concrete_pcca_model)


  return model


def fit_pcca_model(model, x1_train, x2_train, latent_dim, stddv_datapoints):

  data_dim_x1, num_datapoints = x1_train.shape
  data_dim_x2 = x2_train.shape[0]


  z1 = tf.Variable(tf.random.normal([latent_dim, num_datapoints]))
  z2 = tf.Variable(tf.random.normal([latent_dim, num_datapoints]))
  zshared = tf.Variable(tf.random.normal([latent_dim, num_datapoints]))
  lambda1 = tf.Variable(tf.random.normal([data_dim_x1, latent_dim]))
  lambda2 = tf.Variable(tf.random.normal([data_dim_x2, latent_dim]))
  b1 = tf.Variable(tf.random.normal([data_dim_x1, latent_dim]))
  b2 = tf.Variable(tf.random.normal([data_dim_x2, latent_dim]))

  target_log_prob_fn = lambda z1, z2, zshared, lambda1, lambda2, b1, b2: model.log_prob((z1, z2, zshared, lambda1, lambda2, b1, b2, x1_train, x2_train))

  # MAP estimate
  # losses = tfp.math.minimize(
  #     lambda: -target_log_prob_fn(z1, z2, zshared, lambda1, lambda2, b1, b2),
  #     optimizer=tf.optimizers.Adam(learning_rate=0.05),
  #     num_steps=4000)


  qz1_mean = tf.Variable(tf.random.normal([latent_dim, num_datapoints]))
  qz2_mean = tf.Variable(tf.random.normal([latent_dim, num_datapoints]))
  qzshared_mean = tf.Variable(tf.random.normal([latent_dim, num_datapoints]))
  qlambda1_mean = tf.Variable(tf.random.normal([data_dim_x1, latent_dim]))
  qlambda2_mean = tf.Variable(tf.random.normal([data_dim_x2, latent_dim]))
  qb1_mean = tf.Variable(tf.random.normal([data_dim_x1, latent_dim]))
  qb2_mean = tf.Variable(tf.random.normal([data_dim_x2, latent_dim]))

  qz1_stddv = tfp.util.TransformedVariable(1e-4 * tf.ones([latent_dim, num_datapoints]),
                                          bijector=tfb.Softplus())
  qz2_stddv = tfp.util.TransformedVariable(1e-4 * tf.ones([latent_dim, num_datapoints]),
                                          bijector=tfb.Softplus())
  qzshared_stddv = tfp.util.TransformedVariable(1e-4 * tf.ones([latent_dim, num_datapoints]),
                                          bijector=tfb.Softplus())
  qlambda1_stddv = tfp.util.TransformedVariable(1e-4 * tf.ones([data_dim_x1, latent_dim]),
                                          bijector=tfb.Softplus())
  qlambda2_stddv = tfp.util.TransformedVariable(1e-4 * tf.ones([data_dim_x2, latent_dim]),
                                          bijector=tfb.Softplus())
  qb1_stddv = tfp.util.TransformedVariable(1e-4 * tf.ones([data_dim_x1, latent_dim]),
                                          bijector=tfb.Softplus())
  qb2_stddv = tfp.util.TransformedVariable(1e-4 * tf.ones([data_dim_x2, latent_dim]),
                                          bijector=tfb.Softplus())
  def factored_normal_variational_model():
    qz1 = yield tfd.Normal(loc=qz1_mean, scale=qz1_stddv, name="qz1")
    qz2 = yield tfd.Normal(loc=qz2_mean, scale=qz2_stddv, name="qz2")
    qzshared = yield tfd.Normal(loc=qzshared_mean, scale=qzshared_stddv, name="qzshared")
    qlambda1 = yield tfd.Normal(loc=qlambda1_mean, scale=qlambda1_stddv, name="qlambda1")
    qlambda2 = yield tfd.Normal(loc=qlambda2_mean, scale=qlambda2_stddv, name="qlambda2")
    qb1 = yield tfd.Normal(loc=qb1_mean, scale=qb1_stddv, name="qb1")
    qb2 = yield tfd.Normal(loc=qb2_mean, scale=qb2_stddv, name="qb2")

  surrogate_posterior = tfd.JointDistributionCoroutineAutoBatched(
      factored_normal_variational_model)

  losses = tfp.vi.fit_surrogate_posterior(
      target_log_prob_fn,
      surrogate_posterior=surrogate_posterior,
      optimizer=tf.optimizers.Adam(learning_rate=0.05),
      num_steps=1000)



  

  model_dict = {
  'loss_trace': losses,
  'z1': qz1_mean,
  'z2': qz2_mean,
  'zshared': qzshared_mean,
  'lambda1': qlambda1_mean,
  'lambda2': qlambda2_mean,
  'b1': qb1_mean,
  'b2': qb2_mean,
  }

  return model_dict


if __name__ == "__main__":
  num_datapoints = 1000
  latent_dim = 5
  data_dim_x1 = 10
  data_dim_x2 = 10
  stddv_datapoints = 1


  model = pcca_model(data_dim_x1, data_dim_x2, latent_dim, num_datapoints, stddv_datapoints)

  actual_z1, actual_z2, actual_zshared, actual_lambda1, actual_lambda2, actual_b1, actual_b2, x1_train, x2_train = model.sample()

  model_dict = fit_pcca_model(model, x1_train, x2_train, latent_dim, stddv_datapoints)

  # plt.plot(model_dict['loss_trace'])
  # plt.show()

  # ## zx
  # from scipy.stats import pearsonr
  # lv_corrs = np.zeros((latent_dim, latent_dim))
  # for ii in range(latent_dim):
  #   for jj in range(latent_dim):
  #     lv_corrs[ii, jj] = pearsonr(model_dict['lambda1'].numpy()[:, ii], actual_lambda1.numpy()[:, jj])[0]

  # # Greedy approach to ordering
  # order_idx = []
  # for ii in range(lv_corrs.shape[1]):
  #     curr_sorted_idx = np.argsort(-np.abs(lv_corrs[ii, :]))
  #     curr_sorted_idx = curr_sorted_idx[~np.isin(curr_sorted_idx, order_idx)]
  #     curr_argmax = curr_sorted_idx[0]
  #     order_idx.append(curr_argmax)

  # plt.figure(figsize=(21, 6))
  # # plt.subplot(131)
  # sns.heatmap(np.abs(lv_corrs[:, order_idx]), center=0)
  # plt.title("Lambda1")
  # plt.xlabel("True")
  # plt.ylabel("Estimated")
  # plt.show()

  from sklearn.decomposition import PCA
  import pandas as pd
  from scipy.stats import pearsonr

  corrs = np.zeros((latent_dim, latent_dim))
  for ii in range(latent_dim):
    for jj in range(latent_dim):
      corrs[ii, jj] = pearsonr(PCA().fit_transform(actual_lambda1.numpy())[:, ii], PCA().fit_transform(model_dict['lambda1'].numpy())[:, jj])[0]
      # corrs[ii, jj] = pearsonr(PCA().fit_transform(actual_zshared.numpy())[:, ii], PCA().fit_transform(model_dict['zshared'].numpy())[:, jj])[0]
      # corrs[ii, jj] = pearsonr(actual_lambda1.numpy()[:, ii], model_dict['lambda1'].numpy()[:, jj])[0]
      # corrs[ii, jj] = pearsonr(actual_zshared.numpy()[:, ii], model_dict['zshared'].numpy()[:, jj])[0]


  # Greedy approach to ordering
  order_idx = []
  for ii in range(corrs.shape[0]):
      curr_sorted_idx = np.argsort(-np.abs(corrs[ii, :]))
      curr_sorted_idx = curr_sorted_idx[~np.isin(curr_sorted_idx, order_idx)]
      curr_argmax = curr_sorted_idx[0]
      order_idx.append(curr_argmax)

  plt.ylabel("True")
  plt.xlabel("Estimated")

  sns.heatmap(np.abs(corrs[:, order_idx]), center=0)
  plt.show()
  import ipdb; ipdb.set_trace()

  


