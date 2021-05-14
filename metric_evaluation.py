import gpu_util
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_util.pick_gpu_lowest_memory())
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import sys
import skimage.measure
import numpy
import numpy as np
import skimage.io
from skimage.morphology import disk, dilation
import skimage.feature
import skimage.metrics

def compute_metric(dir1, dir2, mode, mask=None, thre=0):

    if mode == 'perceptual_tf':
        sys.path += ['lpips-tensorflow']
        import lpips_tf
        import tensorflow as tf
        image0_ph = tf.placeholder(tf.float32, [1, None, None, 3])
        image1_ph = tf.placeholder(tf.float32, [1, None, None, 3])
        distance_t = lpips_tf.lpips(image0_ph, image1_ph, model='net-lin', net='alex')
        sess = tf.Session()

    files1 = os.listdir(dir1)
    files2 = os.listdir(dir2)
    img_files1 = sorted([file for file in files1 if file.endswith('.png') or file.endswith('.jpg')])
    img_files2 = sorted([file for file in files2 if file.endswith('.png') or file.endswith('.jpg')])
    
    if '--prefix' in sys.argv:
        prefix_idx = sys.argv.index('--prefix')
        prefix = sys.argv[prefix_idx+1]
        img_files2 = [file for file in img_files2 if file.startswith(prefix)]
   
    skip_last_n = 0
    
    
    img_files2 = img_files2
    assert len(img_files1) == len(img_files2)

    vals = numpy.empty(len(img_files1))

    for ind in range(len(img_files1)):
        if mode in ['ssim', 'psnr']:
            img1 = skimage.img_as_float(skimage.io.imread(os.path.join(dir1, img_files1[ind])))
            img2 = skimage.img_as_float(skimage.io.imread(os.path.join(dir2, img_files2[ind])))
            if mode == 'ssim':
                metric_val = skimage.measure.compare_ssim(img1, img2, datarange=img2.max()-img2.min(), multichannel=True)
            elif mode == 'psnr':
                metric_val = skimage.metrics.peak_signal_noise_ratio(img2, img1)
        elif mode == 'perceptual_tf':
            img1 = np.expand_dims(skimage.img_as_float(skimage.io.imread(os.path.join(dir1, img_files1[ind]))), axis=0)
            img2 = np.expand_dims(skimage.img_as_float(skimage.io.imread(os.path.join(dir2, img_files2[ind]))), axis=0)
            metric_val = sess.run(distance_t, feed_dict={image0_ph: img1, image1_ph: img2})
        else:
            raise
        vals[ind] = numpy.mean(metric_val)

    filename_all = mode + '_all.txt'
    filename_breakdown = mode + '_breakdown.txt'
    filename_single = mode + '.txt'
    numpy.savetxt(os.path.join(dir1, filename_all), vals, fmt="%f, ")
    target=open(os.path.join(dir1, filename_single),'w')
    target.write("%f"%numpy.mean(vals))
    target.close()
    
    if len(img_files1) == 30:
        target=open(os.path.join(dir1, filename_breakdown),'w')
        target.write("%f, %f, %f"%(numpy.mean(vals[:5]), numpy.mean(vals[5:10]), numpy.mean(vals[10:])))
        target.close()
    
    if mode == 'perceptual_tf':
        sess.close()
    return vals

def get_score(name):
    dirs = sorted(os.listdir(name))

    if len(sys.argv) > 3:
        mode = sys.argv[3]
    else:
        mode = None
        
    all_modes = ['ssim', 'perceptual_tf', 'psnr']
        
    if mode is None:
        print('running all mode', mode)
        for m in all_modes:
            compute_metric(name, sys.argv[2], m)
    else:
        assert mode in all_modes
        print('running mode', mode)
        compute_metric(name, sys.argv[2], mode)


if __name__ == '__main__':
    get_score(sys.argv[1])
