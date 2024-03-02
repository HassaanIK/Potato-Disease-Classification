# Potato Disease Classification Web App

### Overview
This is a Flask web app that takes an image of a potato leaf as input and predicts whether it's healthy, has early blight, or has late blight. The project uses a deep learning model based on the CNN architecture.

### Steps
1. Data Collection: The dataset used for this project is taken from [Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease).
2. Data Preparation: The classes taken from dataset consists of images of potato leaves labeled as healthy, early blight, or late blight.
3. Model Architecture: The CNN architecture is used for image classification, which consists of 6 convolution layers and 2 fully connected having 26M parameters.
4. Training: The model is trained on the dataset using a combination of techniques such as data augmentation, early stopping, and learning rate scheduling to improve performance.
5. Evaluation: The model is evaluated on a separate validation set to assess its performance in predicting potato leaf diseases, 98.6% of accuracy of which is acheived.
6. Deployment: The trained model is deployed as a Flask web app, allowing users to upload an image of a potato leaf and get the predicted disease class.

### Functions
`predict_mask(image_path, model)`: This function takes the path to an image of a potato leaf and the trained model as input, pre-processes the image, and predicts the disease class (healthy, early blight, or late blight) along with the probability.

### Techniques Used
- `CNN` Architecture: Used for image classification, providing a robust framework for accurate disease classification.
- `Batch Normalization`: Technique used to normalize the activations of each layer, improving the training speed and stability of the model.
- `Max Pooling`: Pooling operation used to downsample the feature maps, reducing the spatial dimensions and extracting the most important features.
- `Dropout`: Regularization technique used in the last layer to prevent overfitting by randomly setting a fraction of the input units to zero during training, forcing the model to learn more robust features.
- `Data Augmentation`: Applied to the training dataset to increase the diversity of images and improve the model's ability to generalize.
- `Early Stopping`: Technique used to prevent overfitting by monitoring the model's performance on a validation set and stopping training when performance starts to degrade.
- `Learning Rate Scheduling`: Technique used to adjust the learning rate during training, allowing the model to converge faster and potentially reach a better solution.

### Usage
1. Clone the repository to your local machine
2. Install the required dependencies:`pip install -r requirements.txt`
3. Run the Flask web app: `python app.py`
4. Open your web browser and go to `http://localhost:5000`.
5. Upload an image of a potato leaf and click the `Predict` button to get the predicted disease class and probability.

### Web App
![Screenshot (30)](https://github.com/HassaanIK/Potato_disease/assets/139614780/94be88d4-5588-4cd2-823f-1b7f4a548728)

   
