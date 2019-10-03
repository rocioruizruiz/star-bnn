star-bnn project
================
Inference implementation of Matthieu Courbariaux MLP Binarized Neural Network using Yaman Umuroglu formulas<br>
(C, python, Linux, VHDL, FPGA).<br>
The dataset shown here is the simple didactic iris dataset (the csv files).<br>
<br>
Installation:<br>
see install-howto.txt (recommended)<br>
or<br>
pip install -r requirements.txt<br>
<br>
Quick start:<br>
$ python mnist.py<br>
  -> generates the nn-binary.bin file<br>
$ g++ bnn.c -lm<br>
$ ./a.out<br>
  -> reads the nn-binary.bin file and generates fpga_bnn.txt<br>
  -> also shows the results of the bnn applied to the iris test dataset<br>
Copy the fpga_bnn.txt contents to inference32.vhd (or inference.vhd) and <br>
synthesize or simulate vhdl code.<br>
VHDL uses floating point half precision units based on the flopoco project<br>
(which has multiple synthesis warnings).<br>
<br>
The neural network structure is specified in mnist.py, bnn.c and float_sizes.vhd.<br>
These values must be identical:
- number of neural network inputs
- number of hidden layers
- number of neurons per hidden layer
- number of neural network outputs
<br>

Please provide feedback!<br>
laurentiu [dot] duca [at] gmail [dot] com
