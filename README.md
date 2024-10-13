# House Price Prediction

Please check out
[House Price Prediction Competition at Kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview).

We are trying to predict house prices on a test set by using provided labeled data.

This GitHub project's results will be upstreamed to `Kaggle` at
[Kaggle Upstream Notebook for this Repository](https://www.kaggle.com/code/fatulm/github-house-price-prediction-upstream).

Here I will fix and preprocess data.
Then by using a deep neural network I will try to impute missing values.
And Finally using a deep neural network I will try to predict prices for the test dataset.

I will use several python packages such as:

- numpy
- pandas
- scipy
- scikit-learn
- joblib
- keras (with jax backend)
- matplotlib, seaborn and plotly (for visualization)

The main codes are in top directory, and should be run sequentially:

- _imports.py
- step_1_fix.py: fixing some problems in input files.
- step_2_preprocess.py: basic preprocessing such as basic imputation, one-hot encoding and standardizing features.
- step_3_impute.py: impute NAs using a deep neural network.
- step_4_predict.py: predict test results using a deep neural network.

The inputs are downloaded from `Kaggle` and saved to `input` folder.

The outputs are saved to `output` folder.

The middle data between input and output (as well as persisted models) are saved to `data` folder.

If you want to get exact same results as me please use conda environment in `environment.yml` file.

There are some ipython notebooks which I used to explore different topics in the `notebooks` folder.

- notebooks/impute-pca.ipynb: Trying to impute NAs by repeatedly using PCA with a low number of components.
- notebooks/keras-autoencoder-pca.ipynb:  Using Keras to build an autoregression model to impute NAs and then using an
  MLP to make prediction.
- notebooks/keras-impute-autor-mlp.ipynb: Using Keras to build an autoencoder in 2 dimensions to visualize data and then
  applying PCA to it.
- notebooks/keras-mlp.ipynb: Using Keras to build an MLP to make prediction.
- notebooks/mlp.ipynb: Using scikit-learn MLP to make prediction.
- notebooks/mlp-feature-importance.ipynb: Finding feature importances using a neural network model trained using Keras.
- notebooks/mlp-learning-rate.ipynb: Finding optimal learning rate for model training.
- notebooks/pca-lda-old.ipynb: Multiple different lower dimension embedding techniques such as PCA and LDA to visualize
  data in 2 and 3 dimensions. This is done when I didn't add ordinal features as ordinarily encoded when preprocessing
  data.
- notebooks/pca-lda.ipynb: Multiple different lower dimension embedding techniques such as PCA and LDA to visualize data
  in 2 and 3 dimensions.
- notebooks/rf-explore.ipynb: Explore different RF and Tree Ensembles on dataset.
- notebooks/rf-lin-cross-val.ipynb: Cross validate different models such as RF and a Regularized Linear model in
  randomized manner.
- notebooks/rf-pred.ipynb: Using RF to make prediction.

This work is to continue my previous work on this dataset which is stored in Kaggle at
[Previous Work at Kaggle about House Price Prediction](https://www.kaggle.com/code/fatulm/house-price-prediction).
Which itself is forked from my friend's notebook on kaggle
[Another Notebook on Kaggle](https://www.kaggle.com/code/mahyarpoorjafary/house-price-prediction).
