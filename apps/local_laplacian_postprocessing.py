import tensorflow as tf
import os
import sys
from local_laplacian_tf import local_laplacian_tf
import skimage.io
import skimage
import numpy
import time

min_res = 64

profile_timing = False
nburns = 10

def main():
    dir = sys.argv[1]
    input = tf.placeholder(tf.float32, [None, None, 3])
    output = local_laplacian_tf(tf.expand_dims(input, axis=0), J=7)
    sess = tf.Session()
    files = os.listdir(dir)
    target_dir = sys.argv[2]
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    
    count = 0
    accum_time = 0
    
    for file in files:
        if file.endswith('.png') and ('local_laplacian' not in file):
            in_img = skimage.img_as_float(skimage.io.imread(os.path.join(dir, file)))
            if in_img.shape[0] % min_res != 0:
                raise
                #dif = int((in_img.shape[0] % min_res) / 2)
                new_res = min_res * int(numpy.floor(in_img.shape[0] / min_res))
                #in_img = in_img[dif:dif+new_res, :, :]
                in_img = in_img[:new_res, :, :]
            if in_img.shape[1] % min_res != 0:
                raise
                #dif = int((in_img.shape[1] % min_res) / 2)
                new_res = min_res * int(numpy.floor(in_img.shape[1] / min_res))
                #in_img = in_img[:, dif:dif+new_res, :]
                in_img = in_img[:, :new_res, :]
                
            T0 = time.time()
            out_img = sess.run(output, feed_dict={input: in_img})
            T1 = time.time()
            count += 1
            
            if count > nburns:
                accum_time += T1 - T0
            
            if not profile_timing:
                out_img = numpy.clip(numpy.squeeze(out_img), 0.0, 1.0)
                skimage.io.imsave(os.path.join(target_dir, 'local_laplacian_' + file), out_img)
                
    return accum_time / (count - nburns)

if __name__ == '__main__':
    accum_time = main()
    print('accum_time:', accum_time)
