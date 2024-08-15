# Hybrid-Net

Real-time audio to chords, lyrics, beat, and melody.

A transformer-based hybrid multimodal model, various transformer models address different problems in the field of music information retrieval, these models generate corresponding information dependencies that mutually influence each other.

An AI-powered multimodal project focused on music, generate chords, beats, lyrics, melody, and tabs for any song.

> The online experience, [See the site here](https://lamucal.com)  

 <img src='./image/tnn.png'  style="width: 750px;" > 
 
<img src='./image/model.png'  style="width: 950px;" >   

`U-Net` network model for audio source separation, `Pitch-Net`, `Beat-Net`, `Chord-Net` and `Segment-Net` based on the transformer model. Apart from establishing the correlation between the frequency and time, the most important aspect is to establish the mutual influence between different networks.   

The entire AI-powered process is implemented in `aitabs.py`, while the various network structure models can be referenced in the `models` folder.   
> **Note**: `U-Net` and `Segment-Net` use the stft spectrum of audio as input. `Beat-Net` uses three spectrograms of drums, bass, and other instruments as input,`Chord-Net` uses one spectrogram of the background music.


## Features
- **Chord**, music chord detection, including major, minor, 7, maj7, min7, 6, m6, sus2, sus4, 5, and inverted chords. Determining the **key** of a song.       

- **Beat**, music beat, downbeat detection and **tempo** (BPM) tracking   

- **Pitch**, tracking the pitch of the melody in the vocal track.  

- **Music Structure**, music segment boundaries and labels, include intro, verse, chorus, bridge and etc.    

- **Lyrics**, music lyrics recognition and automatic lyrics to audio alignment, use ASR (whisper) to recognize the lyrics of the vocal track. The alignment of lyrics and audio is achieved through fine-tuning the wav2vec2 pre-training model. Currently, it supports dozens of languages, including English, Spanish, Portuguese, Russian, Japanese, Korean, Arabic, Chinese, and more.   

- **AI Tabs**, Generate playable sheet music, including chord charts and six-line staves, using chords, beats, music structure information, lyrics, rhythm, etc. It supports editing functionalities for chords, rhythm, and lyrics.   

- **Other**, audio source separation, speed adjustment, pitch shifting, etc.      

For more AI-powered feature experiences, see the [website](https://lamucal.com): 

## Cover
Using a combination of audio STFT, MFCC, and chroma features, with a Transformer model for timbre feature 
modeling and high-level abstraction, this approach can maximize the avoidance of overfitting and underfitting 
problems compared to using a single feature, and has better generalization capabilities. With a small amount of 
data and minimal training, it can achieve better results.

> The online experience, [See the site here](https://lamucal.com/ai-cover) 


<img src='./image/net1.png'  style="width: 950px;" >   
<img src='./image/net2.png'  style="width: 950px;" >   

The model begins by processing the audio signal through a `U-Net`, which isolates the vocal track. 
The vocal track is then simultaneously fed into `PitchNet` and `HuBERT` (Wav2Vec2). `PitchNet` is 
responsible for extracting pitch features, while `HuBERT` captures detailed features of the vocals.

The core of the model is `CombineNet`, which receives features from the `Features` module. This
module consists of three spectrograms: STFT, MFCC, and Chroma, each extracting different aspects
of the audio. These features are enhanced by the TimbreBlock before being passed to the Encoder. 
During this process, noise is introduced via STFT transformation and combined with the features 
before entering the Encoder for processing. The processed features are then passed to the Decoder, 
where they are combined with the previous features to generate the final audio output.

`CombineNet` is based on an encoder-decoder architecture and is trained to generate a mask that 
is used to extract and replace the timbre, ultimately producing the final output audio.

The entire AI-powered process is implemented in `run.py`, while the various network structure 
models can be referenced in the `models` folder.   

## Demo
The results of training on a 1-minute speech of Donald Trump are as follows: 

<table>
<tr>
<td align="center">
  
**Train 10 epoch(Hozier's Too Sweet)**
  
</td>
<td align="center">
  
**Train 100 epoch(Hozier's Too Sweet)**
  
</td>
</tr>
<tr>
<td align="center">

[Train 10 epoch.webm](https://github.com/user-attachments/assets/992747d6-3e47-442c-ab63-0742c83933ee)

</td>
<td align="center">

[Train 100 epoch.webm](https://github.com/user-attachments/assets/877d2cae-d7b7-4355-807f-424ada7df3a1)

</td>
</tr>
</table>


You can experience creating your own voice online, [See the site here](https://lamucal.com/ai-cover)
