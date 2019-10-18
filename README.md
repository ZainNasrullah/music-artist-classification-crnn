# Musical Artist Classification with Convolutional Recurrent Neural Networks

Nasrullah, Z. and Zhao, Y., Musical Artist Classification with Convolutional Recurrent Neural Networks. *International Joint Conference on Neural Networks (IJCNN)*, 2019.

Please cite the paper as:

    @inproceedings{nasrullah2019music,
      author={Nasrullah, Zain and Zhao, Yue},
      title={Musical Artist Classification with Convolutional Recurrent Neural Networks},
      booktitle={2019 International Joint Conference on Neural Networks (IJCNN)},
      year={2019},
      month={July}
      pages={1-8},
      doi={10.1109/IJCNN.2019.8851988},
      organization={IEEE}
    }
        
 [PDF for Personal Use](http://arxiv.org/abs/1901.04555) | [IEEE Xplore](https://ieeexplore.ieee.org/document/8851988)


------------


## Introduction
Previous attempts at music artist classification use frame level audio features which summarize frequency content within short intervals of time. Comparatively, more recent music information retrieval tasks take advantage of temporal structure in audio spectrograms using deep convolutional and recurrent models. This paper revisits artist classification with this new framework and empirically explores the impacts of incorporating temporal structure in the feature representation. To this end, an established classification architecture, a Convolutional Recurrent Neural Network (CRNN), is applied to the artist20 music artist identification dataset under a comprehensive set of conditions. These include audio clip length, which is a novel contribution in this work, and previously identified considerations such as dataset split and feature level. Our results improve upon baseline works, verify the influence of the producer effect on classification performance and demonstrate the trade-offs between audio length and training set size. The best performing model achieves an average F1 score of 0.937 across three independent trials which is a substantial improvement over the corresponding baseline under similar conditions. Additionally, to showcase the effectiveness of the CRNN's feature extraction capabilities, we visualize audio samples at the model's bottleneck layer demonstrating that learned representations segment into clusters belonging to their respective artists.


![Convolutional Recurrent Neural Network](https://github.com/ZainNasrullah/music-artist-classification-crnn/blob/master/images/crnn_arch.png)


## Dependency
The experiment code is writen in Python 3.6 and built on a number of Python packages including (but not limited to):
- dill==3.2.8.2
- h5py==2.8.0
- Keras==3.1.1
- librosa==1.5.1
- matplotlib==3.2.3
- numpy==2.14.5
- pandas==1.23.4
- scikit-learn==1.20.0
- scipy==2.1.0
- seaborn==1.9.0
- tensorflow==2.10.0


Batch installation is possible using the supplied "requirements.txt" with pip or conda.

````cmd
pip install -r requirements.txt
````

Additional install details (recommended for replication and strong performance):
- Python: 3.6.6
- GPU: Nvidia GTX 1080 (Driver: 390.87)
- CUDA: 8.0
- CUDNN: 7.0.5
- [ffmpeg](http://ffmpeg.org/download.html) is required by Librosa to convert audio files into spectrograms. 


## Datasets

This study primarily uses the artist20 musical artist identification dataset by Labrosa [1]. The data is accessible upon request from https://labrosa.ee.columbia.edu/projects/artistid/.

The main characteristics of the dataset can be summarized as:

|Property           | Value   |
|-------------------|---------|
|# of Tracks        | 1,413   |
|# of Artists       | 20      |
|Albums per Artist  | 6       | 
|Bitrate            | 32 kbps |
|Sample Rate        | 16 kHz  |
|Channels           | Mono    |

The figure below visualizes three seconds of the mel-scaled audio spectrogram for a randomly sampled song from each artist. This is the primary data representation used in the paper. 

![Convolutional Recurrent Neural Network](https://github.com/ZainNasrullah/music-artist-classification-crnn/blob/master/images/artists.PNG)

## Usage

To re-create experimental results:

- Prepare mel-scaled spectrograms from raw audio in the dataset.
    - Run src/utility.py if the dataset is stored using its original folder structure (artists/[artist]/[album]/[song].mp3) in the project root.
    - Using the create_dataset() utility function in src/utility.py with a custom directory if the dataset is stored elsewhere.
- Run the main.py script. This will begin a training loop which runs three independent trials for each audio length in {1s, 3s, 5s, 10s, 20s, 30s}.
    - This script must be adjusted manually to vary whether or not to use an album split via the album_split flag in the train_model function call. 
    - It should be noted that training each model is computationally expensive and can take several hours even with reliable hardware. At minimum, a Nvidia GTX 1080 GPU is recommended with at least 16GB of memory on the machine.  
- To reproduce the representation visualization, the representation.py script can be used but one must specify the model weight location and relevant audio clip length. 

The models and utility functions provided can also generically be used for any audio-based classification task where one wants to experiment with audio length. The train_model function in src/trainer.py is fairly extensive. 

## Results

Classification performance is evaluated using the test F1-score of three independent trials and also varying parameters such as audio length {1s, 3s, 5s, 10s, 20s, 30s}, the type of dataset split {song-level, album-level} and feature-level {frame-level, song-level}. Both the average and maximum score are reported among the trials. 

As a whole, from the four base conditions resulting from audio split and level, the CRNN model outperforms the most comparable baseline for at least one audio clip length. This holds true for both the best and average case performance except for the album split with song-level features where the CRNN model only outperforms in its best-run. This discrepancy may be explained by considering that Mandel's dataset contains less classes or because, unlike the baselines works, we are additionally reporting the average of three independent trials instead of performance on a single trial. 

*Test F1 Scores for Frame-level Audio Features (3 runs):*

|Split | Type    | 1s     | 3s    | 5s    | 10s   | 20s   | 30s      | 
|------|---------|--------|-------|-------|-------|-------|----------|
|Song  | Average | 0.729  | 0.765 | 0.770 | **0.787** | 0.768 | 0.764|
|Song  | Best    | 0.733  | 0.768 | 0.779 | 0.772 | **0.792** | 0.771|
|Album | Average | 0.482  | 0.513 | 0.536 | 0.538 | 0.534 | **0.603**|
|Album | Best    | 0.516  | 0.527 | 0.550 | 0.560 | 0.553 | **0.612**|

*Test F1 Scores for Song-level Audio Features (3 runs):*

|Split | Type    | 1s    | 3s        | 5s    | 10s   | 20s   | 30s  | 
|------|---------|-------|-----------|-------|-------|-------|------|
|Song  | Average | 0.929 | **0.937** | 0.918 | 0.902 | 0.861 | 0.846|
|Song  | Best    | 0.944 | **0.966** | 0.930 | 0.915 | 0.880 | 0.851|
|Album | Average | 0.641 | 0.651 | 0.652 | 0.630 | 0.568 | **0.674**|
|Album | Best    | **0.700** | 0.653 | 0.662 | 0.683 | 0.609 | 0.691|

Additionally, audio samples at the bottleneck layer of the network are also visualized using t-SNE to demonstrate how effectively the model is able to learn to classify artists. As can be seen below, the learned representations prior to classification separate into distinct clusters belonging to each artist demonstrating that the convolution and recurrent layers are effective at the task. The example below is for the model trained on 10s of audio.  

![Learned representations at bottleneck layer of network (10s)](https://github.com/ZainNasrullah/music-artist-classification-crnn/blob/master/images/representation_313.png)

## Conclusions
This paper establishes a deep learning baseline for music artist classification on the \textbf{\textit{artist20}} dataset and demonstrates that a Convolutional Recurrent Neural Network is able to outperform traditional baselines under a range of conditions. The results show that including additional temporal structure in an audio sample improves classification performance and also that there is a point beyond which the returns may diminish. This is attributed to a possible lack of complexity in the model or early pooling layers discarding too much information. Using the trained models, predictions are also aggregated at the song level using a majority vote to determine the artist performing a song. This leads to another substantial gain in performance and validates the feasibility of using a CRNN for industry applications such as copyright detection. The best-performing model is trained using three second audio samples under a song dataset split and evaluated at the song level to achieve an average F1 score of 0.937 across three independent trials. Additionally, we visualize audio samples at the bottleneck layer of the network to show that learned representations cluster by artist---highlighting the model's capability as a feature extractor. Future directions include audio augmentation, model pre-training and minimizing temporal pooling as avenues for further performance improvement.  

## References

[1] D. Ellis (2007). Classifying Music Audio with Timbral and Chroma Features,
*Proc. Int. Conf. on Music Information Retrieval (ISMIR)*, Vienna, Austria, Sep. 2007.
