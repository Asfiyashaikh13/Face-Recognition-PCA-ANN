# Face Recognition using PCA and ANN

This project presents a face recognition system built using Principal Component Analysis (PCA) for dimensionality reduction and Artificial Neural Networks (ANN) for classification. The system is capable of identifying known individuals and detecting unknown or imposter faces through a combination of PCA-space distance thresholds and ANN-based softmax confidence scores.

# Project Objectives
> Extract meaningful facial features using PCA (eigenfaces)

> Classify faces using a trained Artificial Neural Network

> Evaluate system accuracy across varying numbers of principal components (k)

> Detect imposter faces using distance-based and confidence-based thresholds

> Visualize classification results and accuracy metrics

# Technologies Used
Python 3.x

Jupyter Notebook

OpenCV: Image loading and preprocessing

NumPy, Pandas: Matrix manipulation and data handling

scikit-learn: Label encoding, train-test splitting, metrics

TensorFlow / Keras: ANN model definition and training

Matplotlib: Visualization of accuracy and predictions

# How the System Works
1. Preprocessing

> Images are converted to grayscale and resized to 128x128 pixels.

> Each image is flattened into a vector, creating a data matrix of shape (mn × p).

2. Principal Component Analysis (PCA)

> Computes the mean image and subtracts it from all samples (mean-centering).

> Calculates the surrogate covariance matrix and performs eigen decomposition.

> The top k eigenvectors are selected to form the basis of the PCA space (eigenfaces).

> Faces are projected onto this lower-dimensional subspace for feature representation.

3. Artificial Neural Network (ANN)

> PCA-transformed features are used to train an ANN model using Keras.

> The network contains hidden layers with ReLU activation, dropout for regularization, and a final softmax layer for classification.

> Accuracy is computed for multiple values of k (e.g., 10 to 60).

4. Imposter Detection

> Imposter images are projected into the same PCA space.

> Predictions are made using the trained ANN model.

> A face is labeled "UNKNOWN" if either:

     Its softmax confidence is below a set threshold (e.g., 0.5), or

     Its PCA-space distance from known faces exceeds a dynamic threshold (mean + 2 * std of training distances).

# Project Files
     # File	                                         Description
    > Face_Recognition_PCA_ANN.ipynb	   Main Jupyter Notebook containing code and results
    > train_indices.csv               	   Indices of training samples for reproducibility
    > test_indices.csv	                  Indices of testing samples
    > accuracy_vs_k.png	                  Graph showing ANN accuracy vs PCA components
    > README.md	                       Project documentation

# How to Run
> Clone or download the repository.

> Install dependencies using:
  pip install -r requirements.txt

> Place the dataset in the required structure.

> Open Face_Recognition_PCA_ANN.ipynb in Jupyter Notebook.

> Run the notebook cell-by-cell to train the ANN and evaluate results.

# Requirements
The following Python libraries are required:
> numpy
> opencv-python
> pandas
> scikit-learn
> matplotlib
> tensorflow
# You can install them using:
  pip install -r requirements.txt
  
# Author
    Asfiya Shaikh
    Bachelor of Computer Applications (BCA)
    Kristu Jayanti College, Bengaluru
    GitHub: github.com/Asfiyashaikh13

