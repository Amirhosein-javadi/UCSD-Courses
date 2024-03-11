#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Imports 
import jax
import jax.numpy as jnp
import numpy as np
import math 
import matplotlib.pyplot as plt 
from transforms3d._gohlketransforms import euler_from_quaternion
from transforms3d.euler import mat2euler
from transforms3d.quaternions import quat2mat
from PIL import Image


# In[ ]:


# Loading data
import pickle
import sys
def read_data(fname):
  d = []
  with open(fname, 'rb') as f:
    if sys.version_info[0] < 3:
      d = pickle.load(f)
    else:
      d = pickle.load(f, encoding='latin1')  # needed for python 3
  return d

dataset= "1"
cfile = "./data/cam/cam" + dataset + ".p"
ifile = "./data/imu/imuRaw" + dataset + ".p"
vfile = "./data/vicon/viconRot" + dataset + ".p"

try:
    camd = read_data(cfile) # cam: (240, 320, 3, n) - ts: (1, n)
except:
    camd = None
imud = read_data(ifile) # vals: (6, m) - ts: (1, m)

try:
    vicd = read_data(vfile) # rots: (3, 3, t) - ts: (1, t)
except:
    vicd = None


# In[ ]:


def find_ground_truth(vicd):
    n_sample = vicd["ts"].shape[1]
    gt_yaw = np.zeros(n_sample)
    gt_pitch = np.zeros(n_sample)
    gt_roll = np.zeros(n_sample)
    vicd_rots = vicd["rots"]
    for n in range(n_sample):
        gt_yaw[n], gt_pitch[n], gt_roll[n] = mat2euler(vicd_rots[:,:,n], 'rzyx')
    gt_yaw = jnp.array(gt_yaw)
    gt_pitch = jnp.array(gt_pitch)
    gt_roll = jnp.array(gt_roll)
    return gt_yaw, gt_pitch, gt_roll

if vicd is not None:
    gt_yaw, gt_pitch, gt_roll = find_ground_truth(vicd)
else: # for test set 
    gt_yaw, gt_pitch, gt_roll = None, None, None


# In[ ]:


if  gt_yaw is not None:
    plt.figure(figsize=(8, 8))
    plt.subplot(3, 1, 1)
    plt.plot(vicd["ts"][0] - vicd["ts"][0,0], gt_yaw * 180 / math.pi)
    plt.ylabel('Yaw')
    plt.ylim([-180,180])
    plt.grid('on')
    plt.title(f'dataset: {dataset}', fontsize=16)
    
    plt.subplot(3, 1, 2)
    plt.plot(vicd["ts"][0] - vicd["ts"][0,0], gt_pitch * 180 / math.pi)
    plt.ylabel('Pitch')
    plt.grid('on')
    plt.ylim([-180,180])
    
    plt.subplot(3, 1, 3)
    plt.plot(vicd["ts"][0] - vicd["ts"][0,0], gt_roll * 180 / math.pi)
    plt.ylabel('Roll')
    plt.grid('on')
    plt.ylim([-180,180]);


# In[ ]:


plt.figure(figsize=(8, 12))

plt.subplot(6, 1, 1)
plt.plot(imud['ts'][0] - imud['ts'][0,0], imud['vals'][0,:])
plt.ylabel('Ax')
plt.grid('on')
plt.title(f'dataset: {dataset}', fontsize=16)

plt.subplot(6, 1, 2)
plt.plot(imud['ts'][0] - imud['ts'][0,0], imud['vals'][1,:])
plt.ylabel('Ay')
plt.grid('on')

plt.subplot(6, 1, 3)
plt.plot(imud['ts'][0] - imud['ts'][0,0], imud['vals'][2,:])
plt.ylabel('Az')
plt.grid('on')

plt.subplot(6, 1, 4)
plt.plot(imud['ts'][0] - imud['ts'][0,0], imud['vals'][4,:])
plt.ylabel('Wx')
plt.grid('on')

plt.subplot(6, 1, 5)
plt.plot(imud['ts'][0] - imud['ts'][0,0], imud['vals'][5,:])
plt.ylabel('Wy')
plt.grid('on')

plt.subplot(6, 1, 6)
plt.plot(imud['ts'][0] - imud['ts'][0,0], imud['vals'][3,:])
plt.ylabel('Wz')
plt.grid('on')
#fig.suptitle('dataset: {dataset}', fontsize=16)

plt.savefig(f'imud_{dataset}.jpg')


# In[ ]:


g_earth = 9.8
static_time_unit = 500

gyros_sensitivity = 3.33 * 180 / math.pi # (mv/degree/second) --> (mv/radian/second)
gyros_scale = 3300 / 1023 / gyros_sensitivity
# improve it later 
W_x_bias = jnp.mean(imud['vals'][4,:static_time_unit]) #- gt_roll[0 :static_time_unit]/ gyros_scale)
W_y_bias = jnp.mean(imud['vals'][5,:static_time_unit]) #- gt_pitch[0 :static_time_unit] / gyros_scale)
W_z_bias = jnp.mean(imud['vals'][3,:static_time_unit]) #- gt_yaw[0 :static_time_unit] / gyros_scale)
print('Gyros scale and bias is: \ngyros_scale:', gyros_scale, '\nW_x_bias:', W_x_bias, '\nW_y_bias:', W_y_bias, '\nW_z_bias:', W_z_bias)

accelerometer_sensitivity = 300 / g_earth # mv/g --> mv/ (m/s^2)
accelerometer_scale = 3300 / 1023 / accelerometer_sensitivity
A_x_bias = jnp.mean(imud['vals'][0, :static_time_unit])
A_y_bias = jnp.mean(imud['vals'][1, :static_time_unit])
A_z_bias = jnp.mean(imud['vals'][2, :static_time_unit]) - (g_earth / accelerometer_scale)
print('\nAccelerometer scale and bias is: \naccelerometer_scale:', accelerometer_scale, '\nA_x_bias:', A_x_bias, '\nA_y_bias:', A_y_bias, '\nA_z_bias:', A_z_bias)


# In[ ]:


# -1 is multiplied due to flipped direction of IMU w.r.t body frame 
scaled_A_x = (jnp.float32(imud['vals'][0,:]) - A_x_bias) * accelerometer_scale
scaled_A_y = (jnp.float32(imud['vals'][1,:]) - A_y_bias) * accelerometer_scale
scaled_A_z = -1 * (jnp.float32(imud['vals'][2,:]) - A_z_bias) * accelerometer_scale

scaled_W_x = (imud['vals'][4,:] - W_x_bias) * gyros_scale
scaled_W_y = (imud['vals'][5,:] - W_y_bias) * gyros_scale
scaled_W_z = (imud['vals'][3,:] - W_z_bias) * gyros_scale

plt.figure(figsize=(10, 15))

plt.subplot(6, 1, 1)
plt.plot(imud['ts'][0] - imud['ts'][0,0], scaled_A_x)
plt.ylabel('Ax')
plt.grid('on')
plt.title(f'dataset: {dataset}', fontsize=16)

plt.subplot(6, 1, 2)
plt.plot(imud['ts'][0] - imud['ts'][0,0], scaled_A_y)
plt.ylabel('Ay')
plt.grid('on')

plt.subplot(6, 1, 3)
plt.plot(imud['ts'][0] - imud['ts'][0,0], scaled_A_z)
plt.ylabel('Az')
plt.grid('on')

plt.subplot(6, 1, 4)
plt.plot(imud['ts'][0] - imud['ts'][0,0], scaled_W_x)
plt.ylabel('Wx')
plt.grid('on')

plt.subplot(6, 1, 5)
plt.plot(imud['ts'][0] - imud['ts'][0,0], scaled_W_y)
plt.ylabel('Wy')
plt.grid('on')

plt.subplot(6, 1, 6)
plt.plot(imud['ts'][0] - imud['ts'][0,0], scaled_W_z)
plt.ylabel('Wz');
plt.grid('on')

plt.savefig(f'imud_normalized_{dataset}.jpg')


# In[ ]:


def quaternion_exp(q):
    w, v = q[0], q[1:]
    norm_v = jnp.sqrt(jnp.dot(v, v))
    if norm_v < 1e-6:
        return jnp.array([jnp.exp(w), 0, 0, 0])
    exp_scalar = jnp.exp(w)
    exp_vector = (jnp.sin(norm_v) / norm_v) * v
    return exp_scalar * np.concatenate(([np.cos(norm_v)], exp_vector))

def quaternion_exp_batch(q):
    w, v = jnp.split(q, (1,), axis=1)
    norm_v = jnp.linalg.norm(v, axis=1, keepdims=True)
    exp_scalar = jnp.exp(w)
    exp_vector = jnp.where(norm_v < 1e-6,
                           jnp.zeros_like(v),
                           (jnp.sin(norm_v) / norm_v) * v)
    exp_q = exp_scalar * jnp.concatenate([jnp.cos(norm_v), exp_vector], axis=1)
    return exp_q

def quaternion_log(q):
    q = q
    w = q[0]
    v = q[1:]
    norm_v = jnp.sqrt(jnp.dot(v, v)) 
    q_norm = jnp.sqrt(jnp.dot(q, q))
    if norm_v < 1e-3:
        return jnp.array([1.0, 0.0, 0.0, 0.0])
    out = jnp.array([jnp.log(q_norm), * (jnp.arccos(w / q_norm) / norm_v) * v])
    return out

def quaternion_log_batch(q):
    w, v = jnp.split(q, (1,), axis=1)
    norm_v = jnp.linalg.norm(v, axis=1, keepdims=True)
    q_norm = jnp.linalg.norm(q, axis=1, keepdims=True)
    mask = norm_v > 1e-3
    q2 = q[mask[:,0]]
    q2_u = q[~mask[:,0]]
    w2, v2 = jnp.split(q2, (1,), axis=1)
    norm_v2 = jnp.linalg.norm(v2, axis=1, keepdims=True)
    q_norm2 = jnp.linalg.norm(q2, axis=1, keepdims=True)
    log_vector = jnp.concatenate([jnp.log(q_norm2), (jnp.arccos(w2 / q_norm2) / norm_v2) * v2], axis=1)
    log_vector = jnp.concatenate([log_vector,jnp.zeros_like(q2_u)],axis=0)    
    return log_vector

def quaternion_inverse(q):
    w, x, y, z = q
    conjugate = jnp.array([w, -x, -y, -z])
    norm_squared = jnp.dot(q, q)
    inverse = conjugate / norm_squared
    return inverse

def quaternion_inverse_batch(q):
    w, x, y, z = jnp.split(q, 4, axis=1)
    conjugate = jnp.concatenate([w, -x, -y, -z], axis=1)
    norm_squared = jnp.sum(q * q, axis=1, keepdims=True)
    inverse = conjugate / norm_squared
    return inverse


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return jnp.array([w, x, y, z])


def quaternion_multiply_batch(quaternions1, quaternions2):
    w1, x1, y1, z1 = jnp.split(quaternions1, 4, axis=1)
    w2, x2, y2, z2 = jnp.split(quaternions2, 4, axis=1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    result = jnp.concatenate([w, x, y, z], axis=1)
    return result

def motion_model(q_t, tau_t, w_t):
    temp = jnp.concatenate([jnp.array([0]), tau_t * w_t / 2])
    out = quaternion_multiply(q_t, quaternion_exp(temp))
    return out 
    
def motion_model_batch(q_t_arr, tau_t_arr, w_t_arr):
    temp = jnp.concatenate([jnp.zeros((w_t_arr.shape[0],1)), tau_t * w_t_arr / 2], axis=1)
    out = quaternion_multiply_batch(q_t_arr, quaternion_exp_batch(temp))
    return out 

def observation_model(q):
    g_arr = jnp.array([0,0,0, -g_earth])
    h = quaternion_multiply(quaternion_multiply(quaternion_inverse(q), g_arr), q)
    return h

def observation_model_batch(q_arr):
    g_arr = jnp.tile(jnp.array([0.0, 0.0, 0.0, -g_earth])[None, :], (q_arr.shape[0], 1))
    h = quaternion_multiply_batch(quaternion_multiply_batch(quaternion_inverse_batch(q_arr), g_arr), q_arr)
    return h


# In[ ]:


# Initialization
q_0 = jnp.array([1, 0, 0, 0])
T = scaled_A_x.shape[0]
q_array = np.zeros((T-1,4))
for t in range(T-1):
    tau_t = imud['ts'][0,t+1] - imud['ts'][0,t]
    w_t = jnp.array([scaled_W_x[t], scaled_W_y[t], scaled_W_z[t]])
    if t == 0:
        q_array[t,:] = motion_model(q_0, tau_t, w_t)
    else:
        q_array[t,:] = motion_model(q_array[t-1,:], tau_t, w_t)

q_array = jnp.array(q_array)


# In[ ]:


def quaternion_to_euler(q_array):
    angles_array = np.zeros((T, 3))
    angles_array[0,:] = euler_from_quaternion(q_0)
    for t in range(T-1):
        angles_array[t+1,:] = euler_from_quaternion(q_array[t]) # roll, pitch, yaw
    return angles_array

def plot_angles_array(q_array, q_array_0=None):
    if vicd is not None:
        min_t = min(vicd["ts"][0,0], imud['ts'][0,0])
    else:
        min_t = imud['ts'][0,0]
    angles_array = quaternion_to_euler(q_array)
    if q_array_0 is not None:
        angles_array_0 = quaternion_to_euler(q_array_0)
    plt.figure(figsize=(8, 8))
    
    plt.subplot(3, 1, 1)
    plt.title(f'dataset: {dataset}', fontsize=16)
    if vicd is not None:
        plt.plot(vicd["ts"][0] - min_t, gt_yaw)
    plt.plot(imud['ts'][0] - min_t, angles_array[:,2])
    if q_array_0 is not None:
        plt.plot(imud['ts'][0] - min_t, angles_array_0[:,2])
    plt.ylim([-math.pi, math.pi])
    plt.ylabel('Yaw')
    if vicd is not None and q_array_0 is not None:
        plt.legend(['Ground Truth', 'Optimized Prediction', 'Initially Predicted'])
    elif vicd is not None and q_array_0 is None:
         plt.legend(['Ground Truth', 'Initially Predicted'])
    elif vicd is None and q_array_0 is not None:
        plt.legend(['Optimized Prediction', 'Initially Predicted'])
    elif vicd is None and q_array_0 is None:
        plt.legend(['Initially predicted'])
    plt.grid('on')
    
    plt.subplot(3, 1, 2)
    if vicd is not None:
        plt.plot(vicd["ts"][0] - min_t, gt_pitch)
    plt.plot(imud['ts'][0] - min_t, angles_array[:,1])
    if q_array_0 is not None:
        plt.plot(imud['ts'][0] - min_t, angles_array_0[:,1])
    plt.ylim([-math.pi, math.pi])
    plt.ylabel('Pitch')
    if vicd is not None and q_array_0 is not None:
        plt.legend(['Ground Truth', 'Optimized Prediction', 'Initially Predicted'])
    elif vicd is not None and q_array_0 is None:
         plt.legend(['Ground Truth', 'Initially Predicted'])
    elif vicd is None and q_array_0 is not None:
        plt.legend(['Optimized Prediction', 'Initially Predicted'])
    elif vicd is None and q_array_0 is None:
        plt.legend(['Initially predicted'])
    plt.grid('on')
    
    plt.subplot(3, 1, 3)
    if vicd is not None:
        plt.plot(vicd["ts"][0] - min_t, gt_roll)
    plt.plot(imud['ts'][0] - min_t, angles_array[:,0])
    if q_array_0 is not None:
        plt.plot(imud['ts'][0] - min_t, angles_array_0[:,0])
    plt.ylim([-math.pi, math.pi])
    plt.ylabel('Roll');
    if vicd is not None and q_array_0 is not None:
        plt.legend(['Ground Truth', 'Optimized Prediction', 'Initially Predicted'])
    elif vicd is not None and q_array_0 is None:
         plt.legend(['Ground Truth', 'Initially Predicted'])
    elif vicd is None and q_array_0 is not None:
        plt.legend(['Optimized Prediction', 'Initially Predicted'])
    elif vicd is None and q_array_0 is None:
        plt.legend(['Initially predicted'])
    plt.grid('on')

plot_angles_array(q_array)
plt.savefig(f'Euler_init_{dataset}.jpg')


# In[ ]:


def quaternion_to_accelerator(q_array):
    acc_array = np.zeros((T, 4))
    acc_array[0,:] = observation_model(q_0)
    for t in range(T-1):   
        acc_array[t+1,:] = observation_model(q_array[t]) # roll, pitch, yaw
    return acc_array

def plot_acc_array(q_array, q_array_0=None):
    acc_array = quaternion_to_accelerator(q_array)
    if q_array_0 is not None:
        acc_array_0 = quaternion_to_accelerator(q_array_0)
        
    plt.figure(figsize=(8, 8))
    plt.subplot(3, 1, 1)
    plt.title(f'dataset: {dataset}', fontsize=16)
    plt.plot(imud['ts'][0,:T] - imud['ts'][0,0], scaled_A_x)
    plt.plot(imud['ts'][0,:T] - imud['ts'][0,0], acc_array[:,1])
    if q_array_0 is not None:
        plt.plot(imud['ts'][0,:T] - imud['ts'][0,0], acc_array_0[:,1])
    if q_array_0 is not None: 
        plt.legend(['Ground Truth', 'Prediction Optimization', 'Initially Predicted'])
    else:
        plt.legend(['Ground Truth', 'Initially Predicted'])
    plt.grid('on')

    
    plt.subplot(3, 1, 2)
    plt.plot(imud['ts'][0,:T] - imud['ts'][0,0], scaled_A_y)
    plt.plot(imud['ts'][0,:T] - imud['ts'][0,0], acc_array[:,2])
    if q_array_0 is not None:
        plt.plot(imud['ts'][0,:T] - imud['ts'][0,0], acc_array_0[:,2])
    if q_array_0 is not None: 
        plt.legend(['Ground Truth', 'Prediction Optimization', 'Initially Predicted'])
    else:
        plt.legend(['Ground Truth', 'Initially Predicted'])
    plt.grid('on')
    plt.subplot(3, 1, 3)
    plt.plot(imud['ts'][0,:T] - imud['ts'][0,0], scaled_A_z)
    plt.plot(imud['ts'][0,:T] - imud['ts'][0,0], acc_array[:,3])
    if q_array_0 is not None:
        plt.plot(imud['ts'][0,:T] - imud['ts'][0,0], acc_array_0[:,3])
    if q_array_0 is not None: 
        plt.legend(['Ground Truth', 'Prediction Optimization', 'Initially Predicted'])
    else:
        plt.legend(['Ground Truth', 'Initially Predicted'])
    plt.grid('on')

plot_acc_array(q_array)
plt.savefig(f'Acc_init_{dataset}.jpg')


# In[ ]:


def cost_function(q_array):
    motion_model_error = 0
    acceleration_error = 0
    for t in range(T-1): 
        tau_t = imud['ts'][0,t+1] - imud['ts'][0,t]
        w_t = jnp.array([scaled_W_x[t], scaled_W_y[t], scaled_W_z[t]])
        if t == 0:
            new_q = motion_model(q_0, tau_t, w_t)
        else: 
            new_q = motion_model(q_array[t-1,:], tau_t, w_t)
        motion_model_error += jnp.sum((2 * quaternion_log(quaternion_multiply(quaternion_inverse(q_array[t,:]), new_q)))**2)
    for t in range(T-1): 
        a_t = jnp.array([0, scaled_A_x[t+1], scaled_A_y[t+1], scaled_A_z[t+1]])
        h_t = observation_model(q_array[t,:])
        acceleration_error += jnp.sum((a_t - h_t)**2)
    out = 0.5 * motion_model_error + 0.5 * acceleration_error 
    return out

def cost_function_batch(q_array):
    motion_model_error = 0
    acceleration_error = 0
    tau_t_arr = imud['ts'][0,1:] - imud['ts'][0,:-1]
    w_t_values = jnp.array([scaled_W_x[:-1], scaled_W_y[:-1], scaled_W_z[:-1]]).T
    new_q_0 = jnp.reshape(motion_model(q_0, tau_t_arr[0], w_t_values[0]), (1,4))
    new_q_arr_t = motion_model_batch(q_array[:-1,], tau_t_arr[1:], w_t_values[1:,:])
    new_q_arr = jnp.concatenate([new_q_0, new_q_arr_t], axis=0)
    motion_model_error = jnp.sum((2 * quaternion_log_batch(quaternion_multiply_batch(quaternion_inverse_batch(q_array), new_q_arr)))**2)
    a_t_arr = jnp.concatenate([jnp.zeros((T-1,1)), jnp.reshape(scaled_A_x[1:], (-1,1)), 
                               jnp.reshape(scaled_A_y[1:], (-1,1)), jnp.reshape(scaled_A_z[1:], (-1,1))], axis=1)
    h_t_arr = observation_model_batch(q_array)
    acceleration_error = jnp.sum((a_t_arr - h_t_arr)**2)
    out = 0.5 * motion_model_error + 0.5 * acceleration_error 
    return out
    
def motion_model_error_func(q_array):
    tau_t_arr = imud['ts'][0,1:] - imud['ts'][0,:-1]
    w_t_values = jnp.array([scaled_W_x[:-1], scaled_W_y[:-1], scaled_W_z[:-1]]).T
    new_q_0 = jnp.reshape(motion_model(q_0, tau_t_arr[0], w_t_values[0]), (1,4))
    new_q_arr_t = motion_model_batch(q_array[:-1,], tau_t_arr[1:], w_t_values[1:,:])
    new_q_arr = jnp.concatenate([new_q_0, new_q_arr_t], axis=0)
    motion_model_error = jnp.sum((2 * quaternion_log_batch(quaternion_multiply_batch(quaternion_inverse_batch(q_array), new_q_arr)))**2)
    return motion_model_error

def acceleration_error_func(q_array):
    a_t_arr = jnp.concatenate([jnp.zeros((T-1,1)), jnp.reshape(scaled_A_x[1:], (-1,1)), 
                               jnp.reshape(scaled_A_y[1:], (-1,1)), jnp.reshape(scaled_A_z[1:], (-1,1))], axis=1)
    h_t_arr = observation_model_batch(q_array)
    acceleration_error = jnp.sum((a_t_arr - h_t_arr)**2)
    return acceleration_error
    
def normalize_rows(matrix):
    row_norms = jnp.linalg.norm(matrix, axis=1, keepdims=True)
    row_norms = jnp.where(row_norms == 0, 1, row_norms)
    normalized_matrix = matrix / row_norms
    return normalized_matrix

def project_grad_tangent(grad, q):
    proj_val = jnp.sum(grad * q, axis= 1)
    proj_vector = q * proj_val[:, np.newaxis]
    proj_grad = grad - proj_vector
    proj_grad_normalized = normalize_rows(proj_grad)
    return proj_grad_normalized

def find_best_alpha(alpha_arr):
    q_array_2 = q_array * jnp.cos(alpha_arr) + proj_grad * jnp.sin(alpha_arr)
    return cost_function(q_array_2)

def find_best_alpha_batch(alpha_arr):
    q_array_2 = q_array * jnp.cos(alpha_arr) + proj_grad * jnp.sin(alpha_arr)
    return cost_function_batch(q_array_2)


# In[ ]:


q_array_0 = q_array


# In[ ]:


q_array = q_array_0
# file_name = f'q_array_dataset_{dataset}_iter_{0}.npy'
# np.save(file_name, q_array)
loss = []
loss.append(cost_function_batch(q_array))
print('initial cost', loss[0])


# In[ ]:


max_iter = 20
initial_lr = 1e-2

for iter in range(1, max_iter):
    lr = initial_lr
    current_cost = loss[-1]
    grad = jax.grad(cost_function_batch)(q_array)
    # add noise for 
    proj_grad = project_grad_tangent(grad, q_array)
    alpha_arr = jnp.zeros((T-1, 1))
    grad_alpha = jax.grad(find_best_alpha_batch)(alpha_arr)
    best_cost = jnp.inf
    while lr > 1e-4:
        alpha_arr_new = alpha_arr - lr * grad_alpha
        new_q_array = q_array * jnp.cos(alpha_arr_new) + proj_grad * jnp.sin(alpha_arr_new) 
        new_cost = cost_function_batch(new_q_array)
        if new_cost < best_cost:
            best_cost = new_cost
            best_lr = lr 
            best_q = new_q_array
            
        lr = lr / 1.2
    
    if best_cost < current_cost:
        loss.append(best_cost)
        q_array = best_q
        print('iter:', iter)
        print('best_cost:', best_cost)
        print('best_lr:', best_lr)
        # file_name = f'q_array_dataset_{dataset}_iter_{iter}.npy'
        # np.save(file_name, q_array)
        
    else:
        print("No place for improvement")
        break

# file_name = f'loss_{dataset}.npy'
# np.save(file_name, jnp.array(loss))


# In[ ]:


plt.plot(loss)
plt.ylabel('loss')
plt.xlabel('iteration')
plt.grid('on')
plt.title(f'dataset: {dataset}', fontsize=16)
plt.savefig(f'loss_{dataset}.jpg')


# In[ ]:


plot_angles_array(q_array, q_array_0)
plt.savefig(f'Euler_Opt_{dataset}.jpg')


# In[ ]:


plot_acc_array(q_array, q_array_0)
plt.savefig(f'Acc_Opt_{dataset}.jpg')


# In[ ]:


def spherical_to_cartesian(longitude, latitude):
    x = np.cos(latitude) * np.cos(longitude)
    y = np.cos(latitude) * np.sin(longitude)
    z = np.sin(latitude)
    out = np.concatenate((x[:, :, np.newaxis], y[:, :, np.newaxis], z[:, :, np.newaxis]), axis=2)
    return out

def spherical_to_pixel(s_array, panorama_rows, panorama_cols):
    W_pixels = np.int32((panorama_cols/(2 * math.pi)) * (-s_array[:,0] + math.pi))
    H_pixels = np.int32((panorama_rows/(2 * math.pi)) * (-2 * s_array[:,1] + math.pi))
    Output_index = np.concatenate((H_pixels[:, np.newaxis], W_pixels[:, np.newaxis]), axis=1)
    return Output_index
    
def find_nearest(array, x):
    absolute_diff = np.abs(array - x)
    index = np.argmin(absolute_diff)  
    return index

def Create_panorma(method='GT', masking=True):
    assert method in ['GT', 'Optimizatition']
    fov_horizontal = 60 / 180 * math.pi
    fov_vertical = 45 / 180 * math.pi
    rows, cols, channel, num_images = camd['cam'].shape
    delta_lambda = fov_horizontal / cols 
    delta_phi = fov_vertical / rows
    longitude = np.zeros((rows, cols))
    latitude = np.zeros((rows, cols))
    for i in range(cols):
        for j in range(rows):
            longitude[j, i] = fov_horizontal / 2 - i * delta_lambda
            latitude[j, i] = fov_vertical / 2 - j * delta_phi
    spherical_coordinates = np.concatenate((longitude[:, :, np.newaxis], latitude[:, :, np.newaxis], np.ones((rows, cols, 1))), axis=2) # (λ,ϕ,1)
    cartesian_coordinates = spherical_to_cartesian(longitude, latitude) #(x, y, z)
    reshaped_cartesian_coordinates = np.reshape(cartesian_coordinates, (-1, 3)).T
    panorama_rows = 1000
    panorama_cols = 2000
    panorama_image = np.zeros((panorama_rows, panorama_cols, 3))
    for n in range(num_images):
        camera_time = camd['ts'][0,n]
        if method == 'GT':
            best_time_vicd = find_nearest(vicd['ts'][0], camera_time)
            R0 = vicd['rots'][:,:,best_time_vicd]
        elif method == 'Optimizatition':   
            best_time_imud = find_nearest(imud['ts'][0], camera_time) - 1
            R0 = quat2mat(q_array[best_time_imud])
        rotated_cartesian_coordinates = R0 @ reshaped_cartesian_coordinates
        # rotated_cartesian_coordinates[2,:] += 0.1
        new_latitude = np.arcsin(rotated_cartesian_coordinates[2,:])
        new_longitude = np.arctan2(rotated_cartesian_coordinates[1,:], rotated_cartesian_coordinates[0,:])
        new_spherical_coordinates = np.concatenate((new_longitude[:, np.newaxis], new_latitude[:, np.newaxis], np.ones((rows * cols, 1))), axis=1)
        index = spherical_to_pixel(new_spherical_coordinates, panorama_rows, panorama_cols)
        
        image_n = camd['cam'][:,:,:,n]
        image_n_reshape = np.reshape(image_n, (-1,3))
        if masking:
            mask = (panorama_image>0)
            new_image = np.zeros_like(panorama_image)
            new_image[index[:, 0], index[:, 1]] = image_n_reshape
            panorama_image = panorama_image * mask + new_image * (1 - mask)
        else:
            panorama_image[index[:, 0], index[:, 1]] = image_n_reshape
    return panorama_image
    


# In[ ]:


if camd is not None:
    if vicd is not None:
        panorama_image_1 = Create_panorma('GT', masking=False)
    panorama_image_2 = Create_panorma('Optimizatition', masking=False)


# In[ ]:


if camd is not None and vicd is not None:
    print('Panorama Image with Rotations from VICON')
    plt.imshow(np.int32(panorama_image_1))
    plt.axis('off') 
    plt.show()
    panorama_image_pill = Image.fromarray(np.uint8(panorama_image_1), 'RGB')
    output_path = f'panorama_image_dataset_{dataset}_GT.jpg'
    panorama_image_pill.save(output_path)


# In[ ]:


if camd is not None:
    print('Panorama Image with Rotations from optimizations')
    plt.imshow(np.int32(panorama_image_2))
    plt.axis('off') 
    plt.show()
    panorama_image_pill = Image.fromarray(np.uint8(panorama_image_2), 'RGB')
    output_path = f'panorama_image_dataset_{dataset}_Optimizatition.jpg'
    panorama_image_pill.save(output_path)


# In[ ]:





# In[ ]:





# In[ ]:




