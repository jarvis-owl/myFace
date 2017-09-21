# myFace

This project finally shall include a trainable convolutional network to recognize familiar faces (distinguish them).

## Part 1 - Video

- capture - __how to collect ?__

    any video? write android app?
    
- name/tag
- crop

  A pretrained face-finder might become handy.
  Use 'Haar-Cascade-Classifier' or ___'openface'__, or somekind of googles 'inception'

Openface would be really cool to use - mighty face detection. 
Eventually, it's useful to analyze each frame already here and just store tagged images of cropped detected faces.
(*opencv's annotate* does that manually and writes face positions in images to a file.)
Openface is probably able to cover this whole project by itself, but for now I failed to use it in anyway.
So a Cascade Classifier might be sufficient (not for lateral faces without an extra classifier), what i'd like to cover in some way.
 
## Part 2 - CNN
- load videofiles (at least two different classes to distinguish) and grab tagged frames
- train CNN
- Hyperparameters:
  * layers
  * nodes
  * filter/kernel size
  * stride
  * pooling

- save *.ckpt

## Part 3 - Model Usage

- load model \*.ckpt and feed in image to determine

show video or just return a string?
