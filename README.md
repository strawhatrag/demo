# Tomato Ripeness Classification

This project is an application that uses a machine learning model to classify tomatoes as ripe or unripe based on images. The code is written in Python and uses TensorFlow for model creation, training, and evaluation.

## Prerequisites

- Python 3.8 or later
- The necessary packages listed in the `requirements.txt` file

## Installation

1. Clone this repository:

   ```plaintext
   git clone https://github.com/your-repo-url.git
   ```

2. Navigate to the project directory:

   ```plaintext
   cd your-repo-url
   ```

3. Install the required packages:

   ```plaintext
   pip install -r requirements.txt
   ```

## Dataset

- The dataset should be organized with ripe and unripe tomato images in the `dataset/Images` directory.
- Ripe tomato images should be named as `Riped tomato_*.jpeg` and unripe tomato images should be named as `unriped tomato_*.jpeg`.

## Usage

1. Run the script to train and evaluate the model:

   ```plaintext
   python your_script.py
   ```

2. The script will load the data, create the model, train it, and evaluate it on the test data.

3. The trained model will be saved as `tomato_model.h5`.

## Results

After running the script, the results including accuracy, precision, recall, and a confusion matrix will be printed to the console.

## License

This project is open-source and available under the MIT License.

## Author

- [Your Name](https://your-profile-url)

## Acknowledgements

This project uses TensorFlow for model creation and training and OpenCV for image processing.
