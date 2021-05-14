from render_util import get_shader_dirname
import glob
import os
import numpy
import numpy as np
import skimage
import skimage.io
import scipy.ndimage
import sys
sys.path += ['..']
import argparse_util

tolerance = 2.0

dtype = 'float32'

lo_pct = 20
hi_pct = 80

# Make a simplified normalization assumption
# Once we use enough data points to roughly estimate normalization parameters
# Store these parameters and directly apply to other datasets

def normalize(X_train, feature_bias=None, feature_scale=None):
    
    global lo_pct, hi_pct
    
    if feature_bias is None:
        feature_bias = numpy.zeros(X_train.shape[-1], dtype)
        feature_scale = numpy.zeros(X_train.shape[-1], dtype)

    global tolerance


    print(X_train.shape[-1])


    for m in range(X_train.shape[3]):

        sorted_arr = numpy.sort(X_train[..., m], axis=None)
        sorted_arr = sorted_arr[numpy.isnan(sorted_arr) == 0]
        sorted_arr = sorted_arr[numpy.isinf(sorted_arr) == 0]
        print("Sorted", m)
        min_val = sorted_arr[0]
        max_val = sorted_arr[-1]
        epsilon = min(1e-8 * (max_val - min_val), 1e-8)
        print("epsilon", epsilon)

        finite = 10
        cluster_count = 0
        current_min = min_val
        apply_outlier = True

        for iter in range(finite):
            ind = numpy.searchsorted(sorted_arr, current_min + epsilon, side='right')
            if ind == sorted_arr.shape[0]:
                cluster_count = iter + 1
                apply_outlier = False
                break
            else:
                current_min = sorted_arr[ind]
                print(sorted_arr[ind])

        print("calculating bias and scale to do 0-1 scaling")
        if apply_outlier:
            print("outlier detection")
            Q1 = sorted_arr[int(lo_pct / 100 * (sorted_arr.shape[0] - 1))]
            Q3 = sorted_arr[int(hi_pct / 100 * (sorted_arr.shape[0] - 1))]
            IQR = Q3 - Q1
            if IQR == 0:
                Q1 = min_val
                Q3 = max_val
                IQR = max_val - min_val
                print("IQR=0, does not apply clipping")
            else:
                min_val = max(min_val, Q1 - tolerance * IQR)
                max_val = min(max_val, Q3 + tolerance * IQR)
                print("clipping outlier")
        else:
            print("finite set", cluster_count)
            Q1 = min_val
            Q3 = max_val
            IQR = max_val - min_val

        tiny = numpy.finfo(dtype).tiny
        feature_bias[m] = -min_val
        diff = max_val - min_val
        feature_scale[m] = 1.0/(diff) if diff >= tiny else 1.0
        print("normalized")

    return

def read_filename(filename):

    d = numpy.load(filename)

    nfeatures = d.shape[0]
    ans = d.reshape([nfeatures, 1, d.shape[1], d.shape[2]])

        
    ans = numpy.moveaxis(ans, [0, 1, 2, 3], [3, 2, 0, 1])
    ans = ans.reshape([ans.shape[0], ans.shape[1], nfeatures])

    return ans

def load(subdir, filename):
    ans = numpy.asarray(read_filename(os.path.join(subdir, filename)), dtype)
    
    nan_num = numpy.sum(numpy.isnan(ans))
    inf_num = numpy.sum(numpy.isinf(ans))
    if nan_num > 0 or inf_num > 0:
        print(filename, nan_num, inf_num)

    return ans


def main():
    parser = argparse_util.ArgumentParser(description='PreprocessRawData')
    parser.add_argument('--base_dirs', dest='base_dirs', default='', help='dirs to read files from')
    parser.add_argument('--shadername', dest='shadername', default='render_zigzag', help='shader name to evaluate on')
    parser.add_argument('--geometry', dest='geometry', default='plane', help='geometry to evaluate on')
    parser.add_argument('--lo_pct', dest='lo_pct', type=int, default=25, help='set the low percentile in outlier removal')


    args = parser.parse_args()

    # color channels are not normalized
    base_dirs = args.base_dirs
    shader_name = args.shadername
    geometry = args.geometry
    normal_map = 'none'
    
    global lo_pct, hi_pct
    
    lo_pct = args.lo_pct
    hi_pct = 100 - args.lo_pct




    base_dirs = base_dirs.split(',')


    all_n = []
    features = []
    dirs = []
    
    for base_dir in base_dirs:
        dirname = get_shader_dirname(base_dir, shader_name, normal_map, geometry, render_prefix=True)

        n = len(glob.glob(os.path.join(dirname, 'g_intermediates*.npy')))
        print(n)
        all_n.append(n)
        dirs.append(dirname)
        if not geometry.startswith('boids'):
            for i in range(n):
                try:
                    feature = load(dirname, 'g_intermediates%05d.npy'%(i))
                except:
                    print(dirname, i)
                    raise
                print(i, feature.shape)

                features.append(feature)
        else:
            features.append(np.load(os.path.join(dirname, 'g_intermediates.npy')))


    features = numpy.asarray(features)

    print('finished reading features')


    feature_bias = numpy.zeros(features.shape[-1], dtype)
    feature_scale = numpy.zeros(features.shape[-1], dtype)

    normalize(features, feature_bias=feature_bias, feature_scale=feature_scale)

    print(os.path.join(dirs[0], 'feature_bias_%d_%d%s.npy' % (lo_pct, hi_pct, '')))
    numpy.save(os.path.join(dirs[0], 'feature_bias_%d_%d%s.npy' % (lo_pct, hi_pct, '')), feature_bias)
    numpy.save(os.path.join(dirs[0], 'feature_scale_%d_%d%s.npy' % (lo_pct, hi_pct, '')), feature_scale)


if __name__ == '__main__':
    main()
