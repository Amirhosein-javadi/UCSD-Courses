The code is designed to be highly adaptable, allowing for easy execution with any dataset. Simply adjust the dataset variable at the beginning of the code to select the desired dataset, with the first nine datasets designated for training and the last two for testing.
Two methods are provided for creating panoramas:
1) Placing each projected image into the panorama (default)
2) Modifying only the zero elements of the panorama image with the projected image.\
To select the second method, simply pass masking=True to the Create_panorama function.
The implementation relies on several libraries, including jax, transforms3d, math, matplotlib, PIL, pickle, and sys.
Before running the code, ensure that you create a data folder in the directory of the .py file and place the camera, IMU, and Vicon datasets within it. For the test set, the datasets should be placed in the same folders as the training set. This organizational structure facilitates easy access and execution of the code with different datasets.