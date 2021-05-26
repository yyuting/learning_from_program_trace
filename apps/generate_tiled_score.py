"""
Saliency code adapted from
https://github.com/immortal3/MLNet-Pytorch
"""

import sys
import os
import numpy as np
import numpy
import skimage
import skimage.io
import skimage.transform

import torch
import torch.nn as nn
import torchvision.models as models

import time
import torchvision.transforms as transforms

import skimage.feature
import skimage.morphology

import cv2

MLNet_model_path = ''


class MLNet(nn.Module):
    
    def __init__(self,prior_size):
        super(MLNet, self).__init__()
        # loading pre-trained vgg16 model and         
        # removing last max pooling layer
        features = list(models.vgg16(pretrained = True).features)[:-1]
        
        # making same spatial size
        # by calculation :) 
        # in pytorch there was problem outputing same size in maxpool2d
        features[23].stride = 1
        features[23].kernel_size = 5
        features[23].padding = 2
                
        self.features = nn.ModuleList(features).eval() 
        # adding dropout layer
        self.fddropout = nn.Dropout2d(p=0.5)
        # adding convolution layer to down number of filters 1280 ==> 64
        self.int_conv = nn.Conv2d(1280,64,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pre_final_conv = nn.Conv2d(64,1,kernel_size=(1, 1), stride=(1, 1) ,padding=(0, 0))
        # prior initialized to ones
        self.prior = nn.Parameter(torch.ones((1,1,prior_size[0],prior_size[1]), requires_grad=True))
        
        # bilinear upsampling layer
        self.bilinearup = torch.nn.UpsamplingBilinear2d(scale_factor=10)
        
    def forward(self, x):
        
        results = []
        for ii,model in enumerate(self.features):
            x = model(x)
            if ii in {16,23,29}:
                results.append(x)
        
        # concat to get 1280 = 512 + 512 + 256
        x = torch.cat((results[0],results[1],results[2]),1) 
        
        # adding dropout layer with dropout set to 0.5 (default)
        x = self.fddropout(x)
        
        # 64 filters convolution layer
        x = self.int_conv(x)
        # 1*1 convolution layer
        x = self.pre_final_conv(x)
        
        upscaled_prior = self.bilinearup(self.prior)
        # print ("upscaled_prior shape: {}".format(upscaled_prior.shape))

        # dot product with prior
        x = x * upscaled_prior
        x = torch.nn.functional.relu(x,inplace=True)
        return x
    
# Modified MSE Loss Function
class ModMSELoss(torch.nn.Module):
    def __init__(self,shape_r_gt,shape_c_gt):
        super(ModMSELoss, self).__init__()
        self.shape_r_gt = shape_r_gt
        self.shape_c_gt = shape_c_gt
        
    def forward(self, output , label , prior):
        prior_size = prior.shape
        output_max = torch.max(torch.max(output,2)[0],2)[0].unsqueeze(2).unsqueeze(2).expand(output.shape[0],output.shape[1],self.shape_r_gt,self.shape_c_gt)
        reg = ( 1.0/(prior_size[0]*prior_size[1]) ) * ( 1 - prior)**2
        loss = torch.mean( ((output / output_max) - label)**2 / (1 - label + 0.1) )  +  torch.sum(reg)
        return loss
    
def comopute_saliency():
    
    # Input Images size
    shape_r = 240
    shape_c = 320
    # shape_r = 480
    # shape_c = 640

    # Output Image size (generally divided by 8 from Input size)
    shape_r_gt = 30
    shape_c_gt = 40
    # shape_r_gt = 60
    # shape_c_gt = 80


    last_freeze_layer = 23
    # last_freeze_layer = 28

    prior_size = ( int(shape_r_gt / 10) , int(shape_c_gt / 10) )

    model = MLNet(prior_size).cuda()


    # freezing Layer
    for i,param in enumerate(model.parameters()):
        if i < last_freeze_layer:
            param.requires_grad = False


    criterion = ModMSELoss(shape_r_gt,shape_c_gt).cuda()

    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3,weight_decay=0.0005,momentum=0.9,nesterov=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-4)

    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    loss_history = []
    nb_epochs = 10
    batch_size = 16

    backup = model.prior
    delattr(model, 'prior')
    model.load_state_dict(torch.load(MLNet_model_path))
    model.prior = backup
    
    src_dir = sys.argv[1]
    prefix = sys.argv[2]
    files = sorted(os.listdir(src_dir))
    files = [file for file in files if file.startswith(prefix)]
    dst_dir = sys.argv[3]
    
    if not os.path.isdir(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)
    
    for i in range(len(files)):
        file = os.path.join(src_dir, files[i])
        if not file.endswith('.png'):
            continue
        img = skimage.img_as_float(skimage.io.imread(file))
        img = skimage.transform.resize(img, (240, 320))
        img_orig = img
        
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        
        out = model.forward(torch.from_numpy(img.astype('f')).cuda())
        
        skimage.io.imsave(os.path.join(dst_dir, '%05d_orig.png' % i), img_orig)
        out_np = out[0].squeeze(0).data.cpu().numpy()
        out_np = skimage.transform.resize(out_np, (240, 320))
        np.save(os.path.join(dst_dir, '%05d_saliency.npy' % i), out_np)
        out_np /= np.max(out_np)
        # saliency is normalized per img for visualization
        # when using saliency to sort tiles should use the raw value from npy files
        skimage.io.imsave(os.path.join(dst_dir, '%05d_saliency.png' % i), out_np)
       
    

                
    
def generate_html():
    dst_dir = sys.argv[3]
    files = sorted(os.listdir(dst_dir))
    files = [file for file in files if file.endswith('.png')]
    
    html_str = """
<html>
<head>
<style>
div.gallery {
border: 1px solid #ccc;
}

div.gallery:hover {
border: 1px solid #777;
}

div.gallery img {
width: 100%;
height: auto;
}

div.desc {
padding: 15px;
text-align: center;
}

* {
box-sizing: border-box;
}

.responsive {
padding: 0 6px;
float: left;
width: 16.666666%;
}

@media only screen and (max-width: 700px) {
.responsive {
width: 49.99999%;
margin: 6px 0;
}
}

@media only screen and (max-width: 500px) {
.responsive {
width: 100%;
}
}

.clearfix:after {
content: "";
display: table;
clear: both;
}
</style>
</head>
<body>
"""
    nfiles = len(files)
    html_str += """
<h2>image files: {nfiles} </h2>
""".format(**locals())

    for i in range(len(files)):
        file = files[i]
        full_filename = file
        html_str += """
<div class="responsive">
<div class="gallery">
<a target="_blank">
<img src="{full_filename}">
</a>
<div class="desc">{file}</div>
</div>
</div>
""".format(**locals())

    html_str += """
<div class="clearfix"></div>
</body>
</html>
"""
    return open(os.path.join(dst_dir, 'index.html'), 'w').write(html_str)

def sort_tiles():
    src_dir = sys.argv[1]
    prefix = sys.argv[2]
    orig_files = sorted(os.listdir(src_dir))
    orig_files = [file for file in orig_files if file.startswith(prefix)]
    dst_dir = sys.argv[3]
    
    if not os.path.isdir(os.path.join(dst_dir, 'tiles')):
        os.mkdir(os.path.join(dst_dir, 'tiles'))
        
    img_tile = 320
    saliency_tile = 80
    all_saliency = np.empty(len(orig_files) * 6)
    
    for i in range(len(orig_files)):
        img_orig = skimage.img_as_float(skimage.io.imread(os.path.join(src_dir, orig_files[i])))
        img_orig = img_orig[160:-160, 160:-160]
        img_saliency = np.load(os.path.join(dst_dir, '%05d_saliency.npy' % i))
        img_saliency = img_saliency[40:-40, 40:-40]
        for tile_h in range(2):
            for tile_w in range(3):
                current_idx = i * 6 + tile_h * 3 + tile_w
                current_tile = img_orig[tile_h*img_tile:(tile_h+1)*img_tile, tile_w*img_tile:(tile_w+1)*img_tile]
                current_saliency = img_saliency[tile_h*saliency_tile:(tile_h+1)*saliency_tile, tile_w*saliency_tile:(tile_w+1)*saliency_tile]
                all_saliency[current_idx] = np.mean(current_saliency)
                skimage.io.imsave(os.path.join(dst_dir, 'tiles', 'tile_%05d_%d_%d.png' % (i, tile_h, tile_w)), current_tile)
                
    np.save(os.path.join(dst_dir, 'tiles', 'tiles_saliency.npy'), all_saliency)
    generate_sort_html(dst_dir, os.path.join(dst_dir, 'tiles', 'tiles_saliency.npy'), 'saliency')
    
def generate_sort_html(dst_dir, metric_file, html_name, metric_idx=None):
    
    html_str = """
<html>
<head>
<style>
div.gallery {
border: 1px solid #ccc;
}

div.gallery:hover {
border: 1px solid #777;
}

div.gallery img {
width: 100%;
height: auto;
}

div.desc {
padding: 15px;
text-align: center;
}

* {
box-sizing: border-box;
}

.responsive {
padding: 0 6px;
float: left;
width: 16.666666%;
}

@media only screen and (max-width: 700px) {
.responsive {
width: 49.99999%;
margin: 6px 0;
}
}

@media only screen and (max-width: 500px) {
.responsive {
width: 100%;
}
}

.clearfix:after {
content: "";
display: table;
clear: both;
}
</style>
</head>
<body>
"""
    all_saliency = np.load(metric_file)
    if metric_idx is not None:
        all_saliency = all_saliency[:, metric_idx]
    
    saliency_sort_idx = np.argsort(all_saliency)[::-1]
    
    for i in range(saliency_sort_idx.shape[0]):
        idx = saliency_sort_idx[i]
        pct = int(i / saliency_sort_idx.shape[0] * 10000) / 100
        img_idx = idx // 6
        tile_h = (idx - img_idx * 6) // 3
        tile_w = idx - img_idx * 6 - tile_h * 3
        imgfile = 'tile_%05d_%d_%d.png' % (img_idx, tile_h, tile_w)
        saliency_score = int(all_saliency[idx] * 1000) / 1000
        if all_saliency[idx] == 0:
            saliency_score = 0
        html_str += """
<div class="responsive">
<div class="gallery">
<a target="_blank">
<img src="{imgfile}">
</a>
<div class="desc">{pct}%, {saliency_score}: {imgfile}</div>
</div>
</div>
""".format(**locals())

    html_str += """
<div class="clearfix"></div>
</body>
</html>
"""
    open(os.path.join(dst_dir, 'tiles', '%s.html' % html_name), 'w').write(html_str)
    
def calc_var(metric):
    dst_dir = sys.argv[3]
    
    assert metric in ['col', 'grad']  or metric.startswith('pyramid')
    
    if metric.startswith('pyramid'):
        sigma, downscale = metric.split('_')[1:]
        sigma = float(sigma)
        if sigma == 0:
            sigma = None
        downscale = int(downscale)
        

    
    all_saliency = np.load(os.path.join(dst_dir, 'tiles', 'tiles_saliency.npy'))
    nimages = all_saliency.shape[0] // 6
    
    if metric.startswith('pyramid'):
        img = skimage.img_as_float(skimage.io.imread(os.path.join(dst_dir, 'tiles', 'tile_%05d_%d_%d.png' % (0, 0, 0))))
        lap_gen = skimage.transform.pyramids.pyramid_laplacian(img, sigma=sigma, downscale=downscale)
        pyramids = [lap for lap in lap_gen]
        all_var = np.empty([all_saliency.shape[0], len(pyramids) * 2])
    else:
        all_var = np.empty(all_saliency.shape)
    
    for i in range(nimages):
        if i % 10 == 0:
            print(i)
        for tile_h in range(2):
            for tile_w in range(3):
                img_ubyte = (skimage.io.imread(os.path.join(dst_dir, 'tiles', 'tile_%05d_%d_%d.png' % (i, tile_h, tile_w))))
                img = skimage.img_as_float(img_ubyte)
                
                idx = i * 6 + tile_h * 3 + tile_w
                
                if metric == 'col':
                    var = np.mean(np.var(img, axis=(0, 1)))
                    feature_mask = None
                elif metric == 'grad':
                    img_gray = skimage.color.rgb2gray(img)
                    laplacian = cv2.Laplacian(img_gray,cv2.CV_64F)
                    var = np.mean(np.abs(laplacian))
                    feature_mask = np.abs(laplacian)
                elif metric.startswith('pyramid'):
                #elif metric == 'pyramid':
                    lap_gen = skimage.transform.pyramids.pyramid_laplacian(img, sigma=sigma, downscale=downscale)
                    pyramids = [lap for lap in lap_gen]
                    var = np.empty(len(pyramids) * 2)
                    feature_mask = np.abs(pyramids[0])
                    
                    for p_ind in range(len(pyramids)):
                        pyramid = pyramids[p_ind]
                        abs_pyramid = np.abs(pyramid)
                        
                        var[p_ind * 2] = np.mean(abs_pyramid)
                        
                        pct_pyramid = np.percentile(abs_pyramid, (92, 94, 96, 98, 100))
                        
                        var[p_ind * 2 + 1] = pct_pyramid[-1]

                all_var[idx] = var
                
                
                
    
    np.save(os.path.join(dst_dir, 'tiles', 'tiles_var_%s.npy' % (metric)), all_var)
    
    if metric.startswith('pyramid'):
        for p_ind in range(all_var.shape[1] // 2):
            generate_sort_html(dst_dir, os.path.join(dst_dir, 'tiles', 'tiles_var_%s.npy' % (metric)), '%s_%d_mean' % (metric, p_ind), metric_idx=p_ind*2)
            generate_sort_html(dst_dir, os.path.join(dst_dir, 'tiles', 'tiles_var_%s.npy' % (metric)), '%s_%d_max' % (metric, p_ind), metric_idx=p_ind*2+1)
    
    else:
        generate_sort_html(dst_dir, os.path.join(dst_dir, 'tiles', 'tiles_var_%s.npy' % (metric)), metric)
    
if __name__ == '__main__':
    
    MLNet_model_path = sys.argv[4]

    comopute_saliency()
    generate_html()
    sort_tiles()
    calc_var('col')
    calc_var('pyramid_0_2')

