star-bnn project
================
Inference implementation of Matthieu Courbariaux MLP Binarized Neural Network using Yaman Umuroglu formulas
(C, python, Linux, VHDL, FPGA).
The dataset shown here is the simple didactic iris dataset (the csv files).

Installation:
see install-howto.txt (recommended)
or
pip install -r requirements.txt

Quick start:
python mnist.py
  -> generates the nn-binary.bin file
g++ bnn.c -lm
./a.out
  -> reads the nn-binary.bin file and generates fpga_bnn.txt
  -> also shows the results of the bnn applied to the iris test dataset
Copy the fpga_bnn.txt contents to inference32.vhd (or inference.vhd) and 
synthesize or simulate vhdl code.
VHDL uses floating point half precision units based on the flopoco project
(which has multiple synthesis warnings).

The neural network structure is specified in mnist.py, bnn.c and float_sizes.vhd.
These values must be identical:
- number of neural network inputs
- number of hidden layers
- number of neurons per hidden layer
- number of neural network outputs

You can find my email by using the following command:
git log -1
Please provide feedback!
