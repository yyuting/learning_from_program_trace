import os
import cv2
import sys
import numpy
import numpy as np
import glob
import skimage
import skimage.io
import shutil
import time

dir = 'mandelbulb_defocus_blur_none_normal_none_test'
kernels = [(3, 3), (5, 5), (9, 9), (17, 17), (33, 33)]
t_dir = '/n/fs/shaderml/1x_1sample_mandelbulb_tile/train/mandelbulb_defocus_blur_none_normal_none'
prefix = 'test_middle_ground'

dir = sys.argv[1]
t_dir = sys.argv[2]
prefix = sys.argv[3]
intermediate_dir = sys.argv[4]
target_img_dir = sys.argv[5]
target_sigma_dir = sys.argv[6]

profile_timing = False

def generate_blurred_imgs():
    files = os.listdir(dir)
    
    accum_time = 0
    file_count = 0
    
    if not os.path.exists(intermediate_dir):
        os.makedirs(intermediate_dir, exist_ok=True)
    
    for file in files:
        if file.startswith(prefix) and file.endswith('.png'):
            imgname, _ = os.path.splitext(file)
            in_img = cv2.imread(os.path.join(dir, file))
            
            T0 = time.time()
            for i in range(len(kernels)):
                kernel = kernels[i]
                sigma = 0.3 * ((kernel[0] - 1) * 0.5 - 1) + 0.8
                blurred_img = cv2.GaussianBlur(in_img, kernel, sigma)
                if not profile_timing:
                    cv2.imwrite(os.path.join(intermediate_dir, imgname + '_blur_%d.png' % i), blurred_img)
            T1 = time.time()
            accum_time += (T1 - T0)
            file_count += 1
            
    return accum_time / file_count

def compute_sigma():
    nframes = len(glob.glob(os.path.join(t_dir, 'g_intermediates*')))
    sigmas = np.empty(len(kernels)+1)
    for i in range(len(kernels)):
        kernel = kernels[i]
        sigmas[i+1] = 0.3 * ((kernel[0] - 1) * 0.5 - 1) + 0.8
    sigmas[0] = 0.0
    sigma_max = sigmas[-1]
    
    accum_time = 0
    
    if not os.path.exists(target_img_dir):
        os.makedirs(target_img_dir, exist_ok=True)
        
    if not os.path.exists(target_sigma_dir):
        os.makedirs(target_sigma_dir, exist_ok=True)

    for i in range(nframes):
        t_ray = numpy.load(os.path.join(t_dir, 'g_intermediates%05d.npy' % i))
        
        T0 = time.time()
        
        t_ray = numpy.squeeze(t_ray)
        t_ray[t_ray <= 0] = numpy.max(t_ray)
        t_ray = cv2.bilateralFilter(t_ray, 0, 15, 15)

        tmax = numpy.max(t_ray)
        tmin = numpy.min(t_ray)
        
        if tmax == tmin and (not profile_timing):
            shutil.copyfile(os.path.join(dir, '%s%05d.png' % (prefix, i)), os.path.join(target_img_dir, 'defocus_blur_%s%05d.png') % (prefix, i))
            numpy.save(os.path.join(target_sigma_dir, 'sigma%s%05d.npy' % (prefix, i)), np.zeros(t_ray.shape))
            continue
        
        t_ray -= tmin
        t_ray /= (tmax - tmin)
        print(i)
        t_ray *= 1.2
        t_ray -= 0.2
        t_ray[t_ray < 0] = 0.0


        current_sigma = t_ray * sigma_max
        
        coefs = [None] * sigmas.shape[0]

        for ind in range(sigmas.shape[0]):
            if ind == 0:
                coefs[ind] = (current_sigma <= sigmas[0]).astype('f') + ((current_sigma > sigmas[0]) * (current_sigma < sigmas[1])).astype('f') * (sigmas[1] - current_sigma) / (sigmas[1] - sigmas[0])
            elif ind == sigmas.shape[0] - 1:
                coefs[ind] = (current_sigma >= sigma_max).astype('f') + ((current_sigma < sigma_max) * (current_sigma > sigmas[ind-1])).astype('f') * (current_sigma - sigmas[ind-1]) / (sigmas[ind] - sigmas[ind-1])
            else:
                coefs[ind] = ((current_sigma <= sigmas[ind]) * (current_sigma > sigmas[ind-1])).astype('f') * (current_sigma - sigmas[ind-1]) / (sigmas[ind] - sigmas[ind-1]) + ((current_sigma > sigmas[ind]) * (current_sigma < sigmas[ind+1])).astype('f') * (sigmas[ind+1] - current_sigma) / (sigmas[ind+1] - sigmas[ind])

            if ind == 0:
                Te0 = time.time()
                img = skimage.img_as_float(skimage.io.imread(os.path.join(dir, '%s%05d.png' % (prefix, i))))
                Te1 = time.time()
                
                ans = img * numpy.expand_dims(coefs[ind], axis=2)
            else:
                Te0 = time.time()
                img = skimage.img_as_float(skimage.io.imread(os.path.join(intermediate_dir, '%s%05d_blur_%d.png' % (prefix, i, ind - 1))))
                Te1 = time.time()
                ans += img * numpy.expand_dims(coefs[ind], axis=2)

        

        # save for range 0 - 1
        current_sigma /= numpy.max(current_sigma)
        
        T1 = time.time()
        accum_time += T1 - T0 - (Te1 - Te0)
        
        if not profile_timing:
            skimage.io.imsave(os.path.join(target_img_dir, 'defocus_blur_%s%05d.png') % (prefix, i), numpy.clip(ans, 0, 1))
            numpy.save(os.path.join(target_sigma_dir, 'sigma%s%05d.npy' % (prefix, i)), current_sigma)
            
    return accum_time / nframes


if __name__ == '__main__':
    accum_time0 = generate_blurred_imgs()
    accum_time1 = compute_sigma()
    print('runtime for 2 phases:', accum_time0, accum_time1)
