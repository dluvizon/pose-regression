## pose-regression - Human pose regression from RGB images

This software implements a human pose regression method based on the Soft-argmax approach, as described in the following paper:
> Human Pose Regression by Combining Indirect Part Detection and Contextual Information ([link](https://arxiv.org/abs/1710.02322))

## Dependencies

The network is implemented using [Keras](https://keras.io/) of top of TensorFlow and Python 3.

We provide a [code](webcan.py) for live demonstration using video frames captured by a webcan. Small changes in the code may be required for hardware compatibility.

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
@article{LUVIZON201915,
title = "Human pose regression by combining indirect part detection and contextual information",
author = "Diogo C. Luvizon and Hedi Tabia and David Picard",
journal = "Computers \& Graphics",
volume = "85",
pages = "15 - 22",
year = "2019",
issn = "0097-8493",
doi = "https://doi.org/10.1016/j.cag.2019.09.002",
}
```

## License

The source code and the weights are given under the MIT License.
