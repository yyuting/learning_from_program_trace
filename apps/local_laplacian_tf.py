import tensorflow as tf
import numpy
import numpy as np
import skimage
import skimage.io
import sys
import numpy.random
import os

os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax([int(x.split()[2]) for x in open('tmp','r').readlines()]))
os.system('rm tmp')

def local_laplacian_tf(input_img, alpha=1.0 / 7.0, beta=1.0, eps=0.01, levels=8, J=8, feed_dict=None):
    # assume input_img = tf.placeholder([1, None, None, 3])
    levels = int(levels)
    J = int(J)

    # gray dim: [1, None, None]
    gray = 0.299 * input_img[:, :, :, 0] + 0.587 * input_img[:, :, :, 1] + 0.114 * input_img[:, :, :, 2]

    gPyramid = [None] * J
    lPyramid = [None] * J
    inGPyramid = [None] * J
    outLPyramid = [None] * J
    outGPyramid = [None] * J
    gPyramid0 = [None] * levels

    for k in range(levels):
        level = k * (1.0 / (levels - 1))
        # idx shape: [1, None, None, 1]
        idx = tf.cast(gray * (levels - 1) * 256, tf.int32)
        idx = tf.clip_by_value(idx, 0, 256 * (levels - 1))
        fx = (tf.cast(idx, tf.float32) - 256.0 * k) / 256.0
        gPyramid0[k] = beta * (gray - level) + level + alpha * fx * tf.exp(-fx * fx / 2.0)
    gPyramid[0] = tf.stack(gPyramid0, axis=3)
    inGPyramid[0] = tf.expand_dims(gray, 3)

    filter_base = numpy.array([[1, 3, 3, 1], [3, 9, 9, 3], [3, 9, 9, 3], [1, 3, 3, 1]], dtype=numpy.float32)
    filter_gPyramid = numpy.zeros([4, 4, levels, levels])
    for i in range(levels):
        filter_gPyramid[:, :, i, i] = filter_base
    filter_inGPyramid = numpy.expand_dims(numpy.expand_dims(filter_base, 2), 3)

    for j in range(1, J):
        gPyramid_old = gPyramid[j-1]
        gPyramid_old_pad = tf.pad(gPyramid_old, [[0, 0], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC')
        gPyramid[j] = tf.nn.conv2d(gPyramid_old_pad, filter_gPyramid, [1, 2, 2, 1], "VALID") / 64.0

        inGPyramid_old = inGPyramid[j-1]
        inGPyramid_old_pad = tf.pad(inGPyramid_old, [[0, 0], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC')
        inGPyramid[j] = tf.nn.conv2d(inGPyramid_old_pad, filter_inGPyramid, [1, 2, 2, 1], "VALID") / 64.0

    lPyramid[J-1] = gPyramid[J-1]

    level = inGPyramid[J-1] * (levels - 1.0)
    li = tf.clip_by_value(tf.cast(level, tf.int32), 0, levels - 2)
    lf = level - tf.cast(li, tf.float32)
    #outLPyramid[J-1] = (1.0 - lf) * tf.gather(lPyramid[J-1], li, axis=3) + lf * tf.gather(lPyramid[J-1], li+1, axis=3)
    meshx, meshy = tf.meshgrid(tf.range(tf.shape(li)[1]), tf.range(tf.shape(li)[2]), indexing='ij')
    #indices = tf.stack([tf.zeros_like(meshx), meshx, meshy, tf.squeeze(li)], axis=2)
    indices = tf.stack([meshx, meshy, tf.squeeze(tf.squeeze(li, axis=3), axis=0)], axis=2)
    indices2 = tf.stack([meshx, meshy, tf.squeeze(tf.squeeze(li+1, axis=3), axis=0)], axis=2)
    outLPyramid[J-1] = (1.0 - lf) * tf.expand_dims(tf.expand_dims(tf.gather_nd(tf.squeeze(lPyramid[J-1], axis=0), indices), axis=0), axis=3) + lf * tf.expand_dims(tf.expand_dims(tf.gather_nd(tf.squeeze(lPyramid[J-1], axis=0), indices2), axis=0), axis=3)

    outGPyramid[J-1] = outLPyramid[J-1]

    filter_base2 = numpy.array([[1, 3], [3, 9]])
    filter_lPyramid = numpy.zeros([2, 2, levels, levels])
    for i in range(levels):
        filter_lPyramid[:, :, i, i] = filter_base2
    filter_outGPyramid = numpy.expand_dims(numpy.expand_dims(filter_base2, axis=2), axis=3)

    for j in range(J - 2, -1, -1):
        #gPyramid_pad = tf.pad(gPyramid[j+1], [[0, 0], [1, 0], [1, 0], [0, 0]], 'SYMMETRIC')
        gPyramid_old = gPyramid[j+1]
        gPyramid_resize = tf.image.resize_images(gPyramid_old, tf.stack([tf.shape(gPyramid_old)[1]*2, tf.shape(gPyramid_old)[2]*2]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        gPyramid_pad = tf.pad(gPyramid_resize, [[0, 0], [1, 0], [1, 0], [0, 0]], 'SYMMETRIC')
        lPyramid[j] = gPyramid[j] - tf.nn.conv2d(gPyramid_pad, filter_lPyramid, [1, 1, 1, 1], "VALID") / 16.0

        level = inGPyramid[j] * (levels - 1.0)
        li = tf.clip_by_value(tf.cast(level, tf.int32), 0, levels - 2)
        lf = level - tf.cast(li, tf.float32)
        meshx, meshy = tf.meshgrid(tf.range(tf.shape(li)[1]), tf.range(tf.shape(li)[2]), indexing='ij')
        indices = tf.stack([meshx, meshy, tf.squeeze(tf.squeeze(li, axis=3), axis=0)], axis=2)
        indices2 = tf.stack([meshx, meshy, tf.squeeze(tf.squeeze(li+1, axis=3), axis=0)], axis=2)
        outLPyramid[j] = (1.0 - lf) * tf.expand_dims(tf.expand_dims(tf.gather_nd(tf.squeeze(lPyramid[j], axis=0), indices), axis=0), axis=3) + lf * tf.expand_dims(tf.expand_dims(tf.gather_nd(tf.squeeze(lPyramid[j], axis=0), indices2), axis=0), axis=3)

        outGPyramid_old = outGPyramid[j+1]
        outGPyramid_resize = tf.image.resize_images(outGPyramid_old, tf.stack([tf.shape(outGPyramid_old)[1]*2, tf.shape(outGPyramid_old)[2]*2]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        outGPyramid_pad = tf.pad(outGPyramid_resize, [[0, 0], [1, 0], [1, 0], [0, 0]], 'SYMMETRIC')
        outGPyramid[j] = outLPyramid[j] + tf.nn.conv2d(outGPyramid_pad, filter_outGPyramid, [1, 1, 1, 1], "VALID") / 16.0

    output = outGPyramid[0] * (input_img + eps) / (tf.expand_dims(gray, axis=3) + eps)
    output = tf.clip_by_value(output, 0.0, 1.0)
    return output
