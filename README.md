## pose-regression - Human pose regression from RGB images

This software implements a human pose regression method based on the Soft-argmax approach, as described in the following paper:
> Human Pose Regression by Combining Indirect Part Detection and Contextual Information ([link](https://arxiv.org/abs/1710.02322))

## Dependencies

The network is implemented using [Keras](https://keras.io/) of top of TensorFlow and Python 3.

We provide a [code](demo_webcan.py) for live demonstration using video frames captured by a webcan. Small changes in the code may be required for hardware compatibility.

The software requires the following packges:

* numpy
* scipy
* keras (2.0 or higher)
* tensorflow (with GPU is better, but is not required)
* pygame (1.9 or higher, only for demonstration)
* matplotlib (only for demonstration)

## Citing

If any part of this source code or the pre-trained weights are useful for you,
please cite the paper:


```
@article{Luvizon_CoRR_2017,
  author    = {Diogo C. Luvizon and Hedi Tabia and David Picard},
  title     = {{Human Pose Regression by Combining Indirect Part Detection and Contextual Information}},
  journal   = {CoRR},
  volume    = {abs/1710.02322},
  year      = {2017},
  url       = {http://arxiv.org/abs/1710.02322},
  archivePrefix = {arXiv},
  eprint    = {1710.02322},
  timestamp = {Wed, 01 Nov 2017 19:05:43 +0100},
  biburl    = {http://dblp.org/rec/bib/journals/corr/abs-1710-02322},
  bibsource = {dblp computer science bibliography, http://dblp.org}
}
```

## License

The source code and the weights are given under the MIT License.
