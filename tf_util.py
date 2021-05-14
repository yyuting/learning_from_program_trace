import tensorflow as tf
from tensor_flow_simplex_matrix import simplex_noise_2arg
import numpy as np
import numpy
import math

dtype = tf.float32

np_perm = numpy.array([151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99,
           37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
           57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48, 27,
           166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102,
           143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116,
           188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126,
           255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152,
           2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224,
           232, 178, 185, 112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81,
           51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50,
           45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61,
           156, 180])

np_grad3_0 = numpy.array([ 1., -1.,  1., -1.,  1., -1.,  1., -1.,  0.,  0.,  0.,  0.])
np_grad3_1 = numpy.array([1.,  1., -1., -1.,  0.,  0.,  0.,  0.,  1., -1.,  1., -1.])
np_grad3_2 = numpy.array([0.,  0.,  0.,  0.,  1.,  1., -1., -1.,  1.,  1., -1., -1.])

def tf_lookup_table_np_perm(x):
    return tf.gather_nd(np_perm, tf.cast(tf.floor(x), tf.int32) % np_perm.shape[0])

def tf_lookup_table_np_grad0(x):
    return tf.gather_nd(np_grad3_0, tf.cast(tf.floor(x), tf.int32) % np_grad3_0.shape[0])

def tf_lookup_table_np_grad1(x):
    return tf.gather_nd(np_grad3_1, tf.cast(tf.floor(x), tf.int32) % np_grad3_1.shape[0])

def tf_lookup_table_np_grad2(x):
    return tf.gather_nd(np_grad3_2, tf.cast(tf.floor(x), tf.int32) % np_grad3_2.shape[0])

def new_mul(x, y):
    try:
        if x == 0.0 or y == 0.0:
            return 0.0
    except:
        pass
    try:
        if (x.dtype == bool) and (y.dtype == bool):
            return tf.logical_and(x, y)
        
        if (x.dtype == bool):
            x = tf.cast(x, y.dtype)
        
        if (y.dtype == bool):
            y = tf.cast(y, x.dtype)
        
    except:
        if x == True:
            return y
        if y == True:
            return x
        if x == False or y == False:
            return False
    return tf.multiply(x, y)


def new_add(x, y):
    if isinstance(x, tf.Tensor) and x.dtype == bool:
        x = tf.cast(x, dtype)
    if isinstance(y, tf.Tensor) and y.dtype == bool:
        y = tf.cast(y, dtype)
    if isinstance(x, bool):
        x = float(x)
    if isinstance(y, bool):
        y = float(y)
    return tf.add(x, y)

tf.Tensor.__mul__ = new_mul
tf.Tensor.__rmul__ = new_mul
tf.Tensor.__add__ = new_add
tf.Tensor.__radd__ = new_add

def select_smooth(a, b, c):
    return a * b + (1 - a) * c

def select_nosmooth(a, b, c):

    all_scalar = True
    base_tensor = None
    count = 0
    for tensor in [a, b, c]:
        if not isinstance(tensor, (int, float, bool, np.bool_)):
            all_scalar = False
            base_tensor = tensor
            count += 1
    if all_scalar:
        return b if a else c

    if base_tensor == a and count == 1:
        actual_dtype = dtype
    elif base_tensor != a:
        actual_dtype = base_tensor.dtype
    else:
        if isinstance(b, tf.Tensor):
            actual_dtype = b.dtype
        else:
            actual_dtype = c.dtype

    if isinstance(b, (int, float, bool)):
        if b == 0.0:
            b = tf.zeros_like(base_tensor, dtype=actual_dtype)
        else:
            if actual_dtype == tf.bool:
                b = tf.ones_like(base_tensor, dtype=actual_dtype)
            else:
                b = b * tf.ones_like(base_tensor, dtype=actual_dtype)
    if isinstance(c, (int, float, bool)):
        if c == 0.0:
            c = tf.zeros_like(base_tensor, dtype=actual_dtype)
        else:
            if actual_dtype == tf.bool:
                c = tf.ones_like(base_tensor, dtype=actual_dtype)
            else:
                c = c * tf.ones_like(base_tensor, dtype=actual_dtype)

        
    return tf.where(tf.cast(a, bool), b, c)

select = select_nosmooth

def simplex_noise(a0, a1, a2, a3, a4, a5, x, y):
    return simplex_noise_2arg(x, y)

def tf_fract(x):
    return tf.floormod(x, 1.0)

def tf_np_wrapper(func):
    def f(x, y=None):
                
        if func == 'sign_up':
            if isinstance(x, tf.Tensor):
                return 2.0 * tf.cast(x >= 0.0, x.dtype) - 1.0
            else:
                return 2.0 * float(x >= 0.0) - 1.0
        elif func == 'sign_down':
            if isinstance(x, tf.Tensor):
                return 2.0 * tf.cast(x > 0.0, x.dtype) - 1.0
            else:
                return 2.0 * float(x > 0.0) - 1.0
        elif func == 'random_normal':
            return tf.random_normal(tf.shape(x), dtype=x.dtype)
        elif func == 'nequal':
            return tf.math.logical_not(tf.equal(x, y))
        elif func == 'expand_1D':
            return tf.expand_dims(x, 1)

        if isinstance(x, (tf.Tensor, tf.Variable)) or isinstance(y, (tf.Tensor, tf.Variable)):
            if func == 'fmod':
                actual_func = tf.floormod
            else:
                actual_func = getattr(tf, func)
        else:
            try:
                actual_func = getattr(np, func)
            except:
                actual_func = getattr(math, func)
        if y is None:
            return actual_func(x)
        else:
            return actual_func(x, y)

    return f

def texture_linear_uv(texture, x, y):
    xx_scaled = x * tf.cast(tf.shape(texture)[1], tf.float32)
    yy_scaled = y * tf.cast(tf.shape(texture)[0], tf.float32)
    
    xx_down = tf.floor(xx_scaled)
    xx_up = xx_down + 1
    yy_down = tf.floor(yy_scaled)
    yy_up = yy_down + 1

    rx = xx_up - xx_scaled
    ry = yy_up - yy_scaled

    xx_up_mod = xx_up
    yy_up_mod = yy_up
    
    val_dd = tf.gather_nd(texture, tf.stack((tf.cast(yy_down, tf.int32), tf.cast(xx_down, tf.int32)), axis=3))
    val_du = tf.gather_nd(texture, tf.stack((tf.cast(yy_up_mod, tf.int32), tf.cast(xx_down, tf.int32)), axis=3))
    val_ud = tf.gather_nd(texture, tf.stack((tf.cast(yy_down, tf.int32), tf.cast(xx_up_mod, tf.int32)), axis=3))
    val_uu = tf.gather_nd(texture, tf.stack((tf.cast(yy_up_mod, tf.int32), tf.cast(xx_up_mod, tf.int32)), axis=3))

    ans = rx * ry * val_dd + rx * (1.0 - ry) * val_du + (1.0 - rx) * ry * val_ud + (1.0 - rx) * (1.0 - ry) * val_uu
    return ans
