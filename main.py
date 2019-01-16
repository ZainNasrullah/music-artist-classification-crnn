# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 13:05:16 2017
Updated on Nov 14 2017
@author: Zain
"""
import os
import pandas as pd
import gc

import src.trainer as trainer

if __name__ == '__main__':

    '''
    1s 32 frames
    3s 94 frames
    5s 157 frames
    6s 188 frames
    10s 313 frames
    20s 628 frames
    29.12s 911 frames
    '''

    slice_lengths = [911, 628, 313, 157, 94, 32]
    random_state_list = [0, 21, 42]
    iterations = 3
    summary_metrics_output_folder = 'trials_song_split'
    for slice_len in slice_lengths:

        scores = []
        pooling_scores = []
        for i in range(iterations):
            score, pooling_score = trainer.train_model(
                nb_classes=20,
                slice_length=slice_len,
                lr=0.001,
                train=True,
                load_checkpoint=True,
                plots=False,
                album_split=False,
                random_states=random_state_list[i],
                save_metrics=True,
                save_metrics_folder='metrics_song_split',
                save_weights_folder='weights_song_split')

            scores.append(score['weighted avg'])
            pooling_scores.append(pooling_score['weighted avg'])
            gc.collect()

        os.makedirs(summary_metrics_output_folder, exist_ok=True)

        pd.DataFrame(scores).to_csv(
            '{}/{}_score.csv'.format(summary_metrics_output_folder, slice_len))

        pd.DataFrame(pooling_scores).to_csv(
            '{}/{}_pooled_score.csv'.format(
                summary_metrics_output_folder, slice_len))
