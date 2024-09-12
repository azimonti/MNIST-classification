# MNIST-classification

Repository demonstrating neural network training for MNIST digit classification.

## Required Tools

- Git
- CMake
- clang
- Fortran
- curl

## Getting Started

To get started with the neural networks:

1. Clone the repository:
   ```
   git clone https://github.com/azimonti/MNIST-classification.git
   ```
2. Navigate to the repository directory:
   ```
   cd MNIST-classification
   ```
3. Initialize and update the submodules:
  ```
  git submodule update --init --recursive
  ```

3. Compile the libraries in `ma_libs`
  ```
  cd externals/ma_libs
  ./cbuild.sh --build-type Debug --cmake-params -DLIBNN=ON
  ./cbuild.sh --build-type Release --cmake-params -DLIBNN=ON
  cd ../..
  ```

  If any error or missing dependencies please look at the instructions [here](https://github.com/azimonti/ma_libs)

4. Dowload the MNIST data (source links were retrieved from the page [here](https://github.com/cvdfoundation/mnist) as the original page gives a 403 Forbidden error), but as far as are the original files and are put in the `data/MNIST/` directory it will be fine.
  ```
  bash download_mnist_data.sh
  ```

5. Compile the binaries
  ```
  ./cbuild.sh -t Release (or -t Debug)
  ```

6. Create the configuation files
  ```
  python nn_config.py
  ```

7. Run the simulations
  ```
  ./build/Release/network1_bin
  ./build/Release/network2_bin --training_start
  ./build/Release/network3_bin --training_start
  ```

## Program 1

The program `network1_bin` is designed to load and display sample MNIST images and their corresponding labels. It uses a simple logging system to either log to the console or to a file based on the `bFileLog` flag.

The program prints a specific image to the console and outputs the corresponding label, providing a visual and numeric understanding of the dataset.

```
Image and label num is: 10000
Image rows: 28, cols: 28
Sample : 36 - Results is: 7
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000000000000000000000000000
0000001111111111111100000000
0000011111111111111110000000
0000011111110000011110000000
0001111111100000001110000000
0001111100000000011110000000
0001110000000000011100000000
0000000000000000111100000000
0000000000000000111000000000
0000000000000001111000000000
0000000000000001111000000000
0000000000000011100000000000
0000000000000111111110000000
0000000000001111111110000000
0000000000001111111110000000
0000000000001111100000000000
0000000000000111000000000000
0000000000000111000000000000
0000000000000111000000000000
0000000000000111000000000000
0000000000000111000000000000
0000000000000000000000000000
0000000100
```

## Program 2

The program `network2_bin` is designed to handle both training and testing of a neural network model using the Stochastic Gradient Descent (SGD) algorithm on the MNIST dataset. It supports the following modes based on command-line arguments:

Command-line Argument Handling:
- `--training_start`: Starts training from scratch.
- `--training_continue`: Continues training from a previously saved state.
- `--testing`: Skips training and directly proceeds with testing the model.

Training and Testing:

- The network trains using the SGD algorithm, where `nepoch` determines the number of epochs (iterations) for training. 
- After training, it can be used for testing on the MNIST dataset.

Performance: The program achieves a good accuracy with 5 epochs, showing more tha 90% accuracy.

## Program 3

The program `network3_bin` uses a Genetic Algorithm (GA) for training a neural network on the MNIST dataset. Like the other programs, it accepts command-line arguments to control whether it should start training from scratch, continue training, or run in testing mode:

Command-line Argument Handling:
- `--training_start`: Starts training from scratch.
- `--training_continue`: Continues training from a previously saved state.
- `--testing`: Skips training and directly proceeds with testing the model.

Training and Testing:

- This program replaces the Stochastic Gradient Descent (SGD) method with a Genetic Algorithm for optimizing the neural network's weights.
- GA Characteristics: Evolutionary processes such as selection, crossover, and mutation are applied across generations to find an optimal set of weights for the network.

Performance: Despite running for 3000 generations, the accuracy remains low, around 30%. This slow convergence is expected because Genetic Algorithms are not well-suited for optimizing neural networks on tasks like MNIST digit classification, where gradient-based methods such as SGD are more effective.

The program not only highlights the limitations of Genetic Algorithms for efficient convergence on tasks like MNIST training but also serves as a demonstration of how to use my library, which can be applied to other tasks more suited for GA or where SGD is not applicable, such as simulating interactions in a game.
