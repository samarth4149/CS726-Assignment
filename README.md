# CS726-Assignment
The final model is saved in the file model.ckpt. The file load_n_test.py is the script that takes in the image as the argument(only 1 argument needed by the script, which is the path to the image), pre-processes it and outputs probability values for the 104 output classes. The file devnflow.py was the one used to train the model and pre-process.py was used to pre-process all training and test images. These scripts need specific folders containing the images to exist in a folder ‘devnagri’, which resides in the same directory as these scripts.

Hence, for the desired output, run:
python load_n_test.py <path_to_test_image>

The code depends on libraries scikit-image, tensorflow and numpy.
