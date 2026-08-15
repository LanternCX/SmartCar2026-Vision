from ctypes import *
import platform
import sys
import os
config_file = 'config.cfg'

def add_post_node(model):
    if sys.platform == 'linux':
        lib = './add_post_processing_64.so'
    elif sys.platform == 'win32':
        arch = platform.architecture()
        if arch[0] == '64bit':
            lib = './add_post_processing_64.dll'
        else:
            lib = './add_post_processing_32.dll'
    lib = CDLL(lib)
    lib.add_node.argtypes = [c_char_p,c_char_p]

    cfg_s = create_string_buffer(256)
    model_s = create_string_buffer(1024)
    cfg_s.raw = bytes(config_file,'utf-8')
    model_s = bytes(model,'utf-8')
    ret = lib.add_node(cfg_s, model_s)

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('-model', help='trained tflite', default=r'final_trained_quant.tflite',type=str)
    args, unknown = parser.parse_known_args()

    add_post_node(args.model)