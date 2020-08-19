# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import glob
import os
import sys
import argparse
import tensorflow as tf

import PIL.Image
import numpy as np
import imageio
from skimage import io
from collections import defaultdict

size_stats = defaultdict(int)
format_stats = defaultdict(int)

def load_image(fname):
    global format_stats, size_stats
    im = imageio.imread(fname)
    # grayscale 
    if len(im.shape) == 3:
        im = np.mean(im, axis=-1)
    
    arr = np.array(im, dtype=np.float32)
    # crop it 
    arr = arr[:256,:256]
    
    # normalize it 
    arr = (arr - arr.min()) / (arr.max() - arr.min()) - 0.5
    
    # add color channel (gray)
    arr = np.expand_dims(arr, axis=2)
    print(arr.shape)
    arr = np.transpose(arr, axes=[2,0,1])
    print(arr.max(), arr.min())
    #assert len(arr.shape) == 3
    return arr


def shape_feature(v):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=v))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

examples='''examples:

  python %(prog)s --input-dir=./kodak --out=imagenet_val_raw.tfrecords
'''

def main():
    parser = argparse.ArgumentParser(
        description='Convert a set of image files into a TensorFlow tfrecords training set.',
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-dir", help="Directory containing ImageNet images")
    parser.add_argument("--out", help="Filename of the output tfrecords file")
    args = parser.parse_args()

    if args.input_dir is None:
        print ('Must specify input file directory with --input-dir')
        sys.exit(1)
    if args.out is None:
        print ('Must specify output filename with --out')
        sys.exit(1)

    print ('Loading image list from %s' % args.input_dir)
    n = 0
    s = glob.glob(args.input_dir+'/train_03/*.jpg')
    images = s[0:50000]


    np.random.RandomState(0x1234f00d).shuffle(images)

    #----------------------------------------------------------
    outdir = os.path.dirname(args.out)
    os.makedirs(outdir, exist_ok=True)
    writer = tf.python_io.TFRecordWriter(args.out)
    for (idx, imgname) in enumerate(images):
        print (idx, imgname)
        image = load_image(imgname)
        feature = {
          'shape': shape_feature(image.shape),
          'data': bytes_feature(tf.compat.as_bytes(image.tostring()))
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    print ('Dataset statistics:')
    print ('  Formats:')
    for key in format_stats:
        print ('    %s: %d images' % (key, format_stats[key]))
    print ('  width,height buckets:')
    for key in size_stats:
        print ('    %s: %d images' % (key, size_stats[key]))
    writer.close()



if __name__ == "__main__":
    main()
