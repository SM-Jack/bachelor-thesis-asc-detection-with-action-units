import os
import numpy as np
import torch
import torch.nn as nn
import logging
import csv
import pandas as pd
from datetime import datetime
from dataset import pil_loader
from utils import *
from conf import get_config,set_logger,set_outdir,set_env

import warnings

# Suppress UserWarning messages
# for registration warning in modeling_finetune.py
warnings.filterwarnings("ignore", category=UserWarning)

def get_prediction(net, img_path):
    net.eval()
    img_transform = image_eval()
    img = pil_loader(img_path)
    img_ = img_transform(img).unsqueeze(0)

    if torch.cuda.is_available():
        net = net.cuda()
        img_ = img_.cuda()

    with torch.no_grad():
        pred = net(img_)
        pred = pred.squeeze().cpu().numpy()

    return pred

def log_old_way(pred):
    dataset_info = hybrid_prediction_infolist # from utils - use hybrid_named_prediction_infolist for intuitive results "lip opener"

    infostr = {'AU prediction:'}
    logging.info(infostr)
    # any AU with value over 0.5 is considered present
    infostr_probs,  infostr_aus = dataset_info(pred, 0.5)
    # which AU are present
    logging.info(infostr_aus)
    # value of each AU
    logging.info(infostr_probs)

    image_prediction = [infostr_aus, infostr_probs]
    return image_prediction

def get_pred_dict(pred):
    pred_dict = create_hybrid_dictionary(pred)

    # Converting set to dictionary
    pred_dict = eval("{"+pred_dict+"}")
    #print(type(pred_dict))
    return pred_dict

def do_single_inference(net, img_path):

    pred = get_prediction(net, img_path)

    return get_pred_dict(pred)

def do_multiple_inference(net, img_path):
    results = {}    # dictionary with each entry being result of one image
    filenames = next(os.walk(img_path), (None, None, []))[2]  # [] if no file
    for img in filenames:
        print(img)
        image_prediction = do_single_inference(net, img_path+img)
        results[str(img)] = image_prediction
    return results

def export_as_csv(pred_dict):
    # where to save
    save_path = "results/video_predictions/"
    # name as date/time
    file_name = "SIT_" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + ".csv"
    save_file_path = save_path + file_name

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(pred_dict, orient='index')

    # save csv
    df.to_csv(save_file_path, index=True)

    return


def main(conf):

    if conf.stage == 1:
        from model.ANFL import MEFARG
        net = MEFARG(num_main_classes=conf.num_main_classes, num_sub_classes=conf.num_sub_classes, backbone=conf.arc, neighbor_num=conf.neighbor_num, metric=conf.metric)
    else:
        from model.MEFL import MEFARG
        net = MEFARG(num_main_classes=conf.num_main_classes, num_sub_classes=conf.num_sub_classes, backbone=conf.arc)
    
    # resume
    if conf.resume != '':
        logging.info("Resume from | {} ]".format(conf.resume))
        net = load_state_dict(net, conf.resume)


    # data
    img_path = conf.input
    pred = {}
    if os.path.isfile(img_path):
        pred = do_single_inference(net, img_path)
    elif os.path.isdir(img_path):
        pred = do_multiple_inference(net, img_path)
    else:
        print("\nFailure: No file or directory as input.\n")
        pred = do_single_inference(net, img_path)

    print("PREDICTION:")
    print(pred)
    print("\n\n")

    export_as_csv(pred)


    # if wanted to draw results on the image itself - irrelevant for us
    if conf.draw_text:
        img = draw_text(conf.input, list(infostr_aus), pred)
        import cv2
        path = conf.input.split('.')[0]+'_pred.jpg'
        cv2.imwrite(path, img)


# ---------------------------------------------------------------------------------

if __name__=="__main__":
    conf = get_config()
    conf.evaluate = True
    set_env(conf)
    # generate outdir name
    set_outdir(conf)
    # Set the logger
    set_logger(conf)
    main(conf)

