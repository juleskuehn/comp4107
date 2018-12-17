# COMP 4107 Final Project (Fall 2018)

  Jules Kuehn, 100661464
  Yunkai Wang, 100968473


The code can also be run on Google Colab. Select "Run all".
https://colab.research.google.com/drive/1UJPnp7TcOuZ55HGt513Q6KsDO69NdVmL
https://colab.research.google.com/drive/1nMMQC2M_AygybKdcORqEF4jUbzyErbH6



## Replicating results of embedding models

Requires the following libraries:
tensorflow, matplotlib, numpy, sklearn, keras

A script is provided to run tests with simple adjustment of model parameters.
The 9 embedding models tested in the paper are commented out.

Running this script will download the datasets from Dropbox (around 50mb for
the book dataset, and 300mb for GloVe embeddings, as required)

```
python run_tests.py
```



## Replicating results of naive models

Requires the following libraries:
tensorflow, matplotlib, numpy, sklearn

To run the tests:
```
python Naive_MLP_Model.py
python Naive_CNN_Model.py
```

Note that the CNN is extremely slow and may appear to be unresposive.
The naive models are not particularly interesting to play with anyway.