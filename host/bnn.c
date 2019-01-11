// author: Laurentiu-Cristian Duca 
// license: GNU GPL
// date: 20181212

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <tensorflow/c/c_api.h>
#include "half.hpp"
using half_float::half;

#define BNN_INPUTS_N 4
//#define BNN_HIDDEN_LAYERS_N 3
#define BNN_HIDDEN_LAYERS_N 2
#define BNN_NEURONS_PER_HIDDEN_LAYER_N 32
#define BNN_OUTPUTS_N 3

// BNN weights and bias
float weights_l1[BNN_INPUTS_N][BNN_NEURONS_PER_HIDDEN_LAYER_N];
float bias_l1[BNN_NEURONS_PER_HIDDEN_LAYER_N];
float weights_lh[BNN_HIDDEN_LAYERS_N - 1][BNN_NEURONS_PER_HIDDEN_LAYER_N][BNN_NEURONS_PER_HIDDEN_LAYER_N];
float bias_lh[BNN_HIDDEN_LAYERS_N - 1][BNN_NEURONS_PER_HIDDEN_LAYER_N];
float weights_lo[BNN_NEURONS_PER_HIDDEN_LAYER_N][BNN_OUTPUTS_N];
float bias_lo[BNN_OUTPUTS_N];

// BNN values
float values_hidden[BNN_HIDDEN_LAYERS_N][BNN_NEURONS_PER_HIDDEN_LAYER_N];
float values_output[BNN_OUTPUTS_N];

// batch normalization
float bnh_beta[BNN_HIDDEN_LAYERS_N][BNN_NEURONS_PER_HIDDEN_LAYER_N], bnh_gamma[BNN_HIDDEN_LAYERS_N][BNN_NEURONS_PER_HIDDEN_LAYER_N], 
	bnh_mean[BNN_HIDDEN_LAYERS_N][BNN_NEURONS_PER_HIDDEN_LAYER_N], bnh_variance[BNN_HIDDEN_LAYERS_N][BNN_NEURONS_PER_HIDDEN_LAYER_N];
float bno_beta[BNN_OUTPUTS_N], bno_gamma[BNN_OUTPUTS_N], bno_mean[BNN_OUTPUTS_N], bno_variance[BNN_OUTPUTS_N];

// thresholds
float th[BNN_HIDDEN_LAYERS_N][BNN_NEURONS_PER_HIDDEN_LAYER_N];
float to[BNN_OUTPUTS_N];

int nInputs, nHiddenLayers, nNeuronsPerHiddenLayer, nOutputs;

#define TRAINING_SAMPLES_NR 120
#define TEST_SAMPLES_NR 30
float training_data[TRAINING_SAMPLES_NR][BNN_INPUTS_N];
float training_outputs[TRAINING_SAMPLES_NR][BNN_OUTPUTS_N];
float test_data[TEST_SAMPLES_NR][BNN_INPUTS_N];
float test_outputs[TEST_SAMPLES_NR][BNN_OUTPUTS_N];
float python_best_test[TEST_SAMPLES_NR][BNN_OUTPUTS_N];
float python_best_idx_test[TEST_SAMPLES_NR];

#define DEBUG_APPLICATION 1
#define debug_printf(pr_str) do { if (DEBUG_APPLICATION) printf pr_str; } while(0) 
#define fgets_dbg(str,x,f) \
    do { \
	if (fgets(str, x, f) == NULL) { \
		printf("%d: fgets==NULL", __LINE__); \
		exit_with_error(); \
	} \
    } \
    while(0)

#define filename "nn-binary.txt"
#define binaryfilename "nn-binary.bin"
#define fpga_bnn_filename "fpga_bnn.txt"
#define iristrainingfilename "iris_training_3_outputs.csv"
#define iristestfilename "iris_test_3_outputs.csv"

FILE *f;
char str[1000];
char sz[20];
char *strPointer;

int exit_with_error() 
{
	fclose(f);
	exit(1);
}

int load_iris_dataset()
{
	int i, j, k, n;
	float max=0;
	FILE *irisfile;

	// training
	printf("Reading file: \"%s\"\n", iristrainingfilename);
	if((irisfile = fopen(iristrainingfilename, "r")) == NULL) {
		printf("Erorr opening \"%s\"", iristrainingfilename);
		exit(1);
	}
	// first line is comment
	fgets_dbg(str, 1000, irisfile);
	for(i = 0; i < TRAINING_SAMPLES_NR; i++) {
		for(j = 0; j < nInputs; j++) {
			fscanf(irisfile, "%f%c", &training_data[i][j], str);
			if(training_data[i][j] > max)
				max = training_data[i][j];
			//printf("%2.1f ", training_data[i][j]);
		}
		for(j = 0; j < nOutputs; j++) {
			fscanf(irisfile, "%f%c", &training_outputs[i][j], str);
			//printf("%2.0f ", training_outputs[i][j]);
		}
		//printf("\n");
	}
	fclose(irisfile);

	// test
	printf("Reading file: \"%s\"\n", iristestfilename);
	if((irisfile = fopen(iristestfilename, "r")) == NULL) {
		printf("Erorr opening \"%s\"", iristestfilename);
		exit(1);
	}
	// first line is comment
	fgets_dbg(str, 1000, irisfile);
	for(i = 0; i < TEST_SAMPLES_NR; i++) {
		for(j = 0; j < nInputs; j++) {
			fscanf(irisfile, "%f%c", &test_data[i][j], str);
			if(test_data[i][j] > max)
				max = test_data[i][j];
			//printf("%2.1f ", test_data[i][j]);
		}
		for(j = 0; j < nOutputs; j++) {
			fscanf(irisfile, "%f%c", &test_outputs[i][j], str);
			//printf("%2.0f ", test_outputs[i][j]);
		}
		//printf("\n");
	}
	fclose(irisfile);

	printf("max=%f\n", max);
	for(i = 0; i < TRAINING_SAMPLES_NR; i++) {
		for(j = 0; j < nInputs; j++) {
			training_data[i][j] /= max;
			training_data[i][j] *= 2;
			training_data[i][j] -= 1;
			//printf("%2.1f ", training_data[i][j]);
		}
		//printf("\n");
	}
	for(i = 0; i < TEST_SAMPLES_NR; i++) {
		for(j = 0; j < nInputs; j++) {
			test_data[i][j] /= max;
			test_data[i][j] *= 2;
			test_data[i][j] -= 1;
			//printf("%2.1f ", test_data[i][j]);
		}
		//printf("\n");
	}
	return 0;
}

void readArrayFromTextFile(int dim1, int dim2, float *arr)
{
	int i, j, k, n;
	float aux;
	// first line is comment
	fgets_dbg(str, 1000, f);
	//debug_printf(("comment='%s'\n", str));
	//printf("dim1=%d dim2=%d\n", dim1, dim2);
	for(i=0; i<dim1; i++) {
		// get rid of [ or ' '
		do {
			fscanf(f, "%c", &str[0]);
			//debug_printf(("c=\"%c\"=%d ", str[0], str[0]));
		} while ((str[0] != '-') && ((str[0] < '0') || (str[0] > '9')));
		//debug_printf(("\n"));
		ungetc(str[0], f);
		for(j=0; j<dim2; j++) {
			fscanf(f, "%f", &aux);
			*(arr + i*dim2 + j) = aux;
			printf("%4.1f ", aux); //*(arr + i*dim2 + j));
		}
		printf("\n");
		// read junk in the rest of the line
		do {
			fscanf(f, "%c", &str[0]);
			if(feof(f)) {
				printf("feof(f) !!!!!!!\n");
				break;
			}
			//debug_printf(("c=\"%c\"=%d \t", str[0], str[0]));
		} while (str[0] != '\n');
		//debug_printf(("\n"));
	}
}

void readVectorFromTextFile(int dim1, float *v)
{
	int i, j, k, n;
	float aux;
	// first line is comment
	fgets_dbg(str, 1000, f);
	//debug_printf(("comment='%s'\n", str));
		// get rid of [ or ' '
		do {
			fscanf(f, "%c", &str[0]);
			//debug_printf(("c=\"%c\"=%d ", str[0], str[0]));
		} while ((str[0] != '-') && ((str[0] < '0') || (str[0] > '9')));
		//debug_printf(("\n"));
		ungetc(str[0], f);
	for(i=0; i<dim1; i++) {
		fscanf(f, "%f", (v + i));
		printf("%4.1f ", *(v + i));
	}
	printf("\n");
		// read junk in the rest of the line
		do {
			fscanf(f, "%c", &str[0]);
			if(feof(f)) {
				printf("feof(f) !!!!!!!\n");
				break;
			}
			//debug_printf(("c=\"%c\"=%d \t", str[0], str[0]));
		} while (str[0] != '\n');
		//debug_printf(("\n"));
}

int readNNFromTextFile() 
{
	int i, j, k, n;
	printf("Reading file: \"%s\"\n", filename);
	if((f = fopen(filename, "r")) == NULL) {
		printf("Erorr opening \"%s\"", filename);
		exit(1);
	}
	// first line is comment
	fgets_dbg(str, 1000, f);
	// second line is nInputs, nHiddenLayers, nNeuronsPerHiddenLayer, nOutputs
	fgets_dbg(str, 1000, f);
	sscanf(str, "%d%d%d%d", &nInputs, &nHiddenLayers, &nNeuronsPerHiddenLayer, &nOutputs);
	printf("nInputs=%d nHiddenLayers=%d nNeuronsPerHiddenLayer=%d nOutputs=%d \n",  nInputs, nHiddenLayers, nNeuronsPerHiddenLayer, nOutputs);
	if((nInputs != BNN_INPUTS_N) || (nHiddenLayers != BNN_HIDDEN_LAYERS_N) || 
		(nNeuronsPerHiddenLayer != BNN_NEURONS_PER_HIDDEN_LAYER_N) || (nOutputs != BNN_OUTPUTS_N)) {
		printf("(nInputs != BNN_INPUTS_N) || (nHiddenLayers != BNN_HIDDEN_LAYERS_N) || "
		"(nNeuronsPerHiddenLayer != BNN_NEURONS_PER_HIDDEN_LAYER_N) || (nOutputs != BNN_OUTPUTS_N) \n");
		fclose(f);
		exit(-1);
	}

	// read layer 1 weights
	printf("weights_l1\n");
	readArrayFromTextFile(nInputs, nNeuronsPerHiddenLayer, (float*)weights_l1);

	// read bias_l1
	printf("bias_l1\n");
	readVectorFromTextFile(nNeuronsPerHiddenLayer, (float*)bias_l1);

	// read weight and bias for other hidden layers
	for(i = 0; i < (nHiddenLayers - 1); i++) {
		printf("nNeuronsPerHiddenLayer=%d nHiddenLayers=%d i=%d\n", nNeuronsPerHiddenLayer, nHiddenLayers, i);
		// read layer (i+2) weights
		printf("weights_l%d\n", (i+2));
		readArrayFromTextFile(nNeuronsPerHiddenLayer, nNeuronsPerHiddenLayer, (float*)&weights_lh[i]);
		// read bias of layer (i+2)
		printf("bias_l%d\n", (i+2));
		readVectorFromTextFile(nNeuronsPerHiddenLayer, (float*)&bias_lh[i]);
	}
	// read layer o weights
	printf("weights_lo\n");
	readArrayFromTextFile(nNeuronsPerHiddenLayer, nOutputs, (float*)weights_lo);
	// read bias_lo
	printf("bias_lo\n");
	readVectorFromTextFile(nOutputs, (float*)bias_lo);

	// read batch normalization for hidden layers
	for(i = 0; i < nHiddenLayers; i++) {
		printf("beta parameters for hidden layer %d (+1):\n", i);
		readVectorFromTextFile(nNeuronsPerHiddenLayer, (float*)&bnh_beta[i][0]);
		printf("gamma parameters for hidden layer %d (+1):\n", i);
		readVectorFromTextFile(nNeuronsPerHiddenLayer, (float*)&bnh_gamma[i][0]);
		printf("mean parameters for hidden layer %d (+1):\n", i);
		readVectorFromTextFile(nNeuronsPerHiddenLayer, (float*)&bnh_mean[i][0]);
		printf("variance parameters for hidden layer %d (+1):\n", i);
		readVectorFromTextFile(nNeuronsPerHiddenLayer, (float*)&bnh_variance[i][0]);
	}
	// read batch normalization for output layer
	printf("beta parameters for output layer\n");
	readVectorFromTextFile(nOutputs, (float*)&bno_beta[0]);
	printf("gamma parameters for output layer\n");
	readVectorFromTextFile(nOutputs, (float*)&bno_gamma[0]);
	printf("mean parameters for output layer\n");
	readVectorFromTextFile(nOutputs, (float*)&bno_mean[0]);
	printf("variance parameters for output layer\n");
	readVectorFromTextFile(nOutputs, (float*)&bno_variance[0]);

	fclose(f);
	return 0;
}

int readNNFromBinaryFile()
{
	int i, j, k, n;
	FILE *fbin=NULL;
	if((fbin = fopen(binaryfilename, "rb")) == NULL) {
		printf("Erorr opening \"%s\"", binaryfilename);	
		exit(-1);
	}

	// Read nInputs, nHiddenLayers, nNeuronsPerHiddenLayer, nOutputs
	// Python saves these values as byte.
	fread(&nInputs, sizeof(char), 1, fbin);
	fread(&nHiddenLayers, sizeof(char), 1, fbin);
	fread(&nNeuronsPerHiddenLayer, sizeof(char), 1, fbin);
	fread(&nOutputs, sizeof(char), 1, fbin);
	printf("nInputs=%d nHiddenLayers=%d nNeuronsPerHiddenLayer=%d nOutputs=%d \n",  nInputs, nHiddenLayers, nNeuronsPerHiddenLayer, nOutputs);
	if((nInputs != BNN_INPUTS_N) || (nHiddenLayers != BNN_HIDDEN_LAYERS_N) || 
		(nNeuronsPerHiddenLayer != BNN_NEURONS_PER_HIDDEN_LAYER_N) || (nOutputs != BNN_OUTPUTS_N)) {
		printf("(nInputs != BNN_INPUTS_N) || (nHiddenLayers != BNN_HIDDEN_LAYERS_N) || "
			"(nNeuronsPerHiddenLayer != BNN_NEURONS_PER_HIDDEN_LAYER_N) || (nOutputs != BNN_OUTPUTS_N) \n");
		fclose(fbin);
		exit(-1);
	}

	// Read NN from file
	// read layer 1 weights
	printf("weights_l1\n");
	for(i = 0; i < nInputs; i++) {
		for(j = 0; j < nNeuronsPerHiddenLayer; j++) {
			fread(&weights_l1[i][j], sizeof(float), 1, fbin);
			printf("%5.2f ", weights_l1[i][j]);
		}
		printf("\n");
	}
	// read bias_l1
	printf("bias_l1\n");
	for(j = 0; j < nNeuronsPerHiddenLayer; j++) {
		fread(&bias_l1[j], sizeof(float), 1, fbin);
		printf("%5.2f ", bias_l1[j]);
	}
	printf("\n");
		printf("Batch normalization parameters for hidden layer %d (+1):\n", 0);
		for(j = 0; j < nNeuronsPerHiddenLayer; j++) {
			fread(&bnh_beta[0][j], sizeof(float), 1, fbin);
			printf("%5.2f ", bnh_beta[0][j]);
		}
		printf("\n");
		for(j = 0; j < nNeuronsPerHiddenLayer; j++) {
			fread(&bnh_gamma[0][j], sizeof(float), 1, fbin);
			printf("%5.2f ", bnh_gamma[0][j]);
		}
		printf("\n");
		for(j = 0; j < nNeuronsPerHiddenLayer; j++) {
			fread(&bnh_mean[0][j], sizeof(float), 1, fbin);
			printf("%5.2f ", bnh_mean[0][j]);
		}
		printf("\n");
		for(j = 0; j < nNeuronsPerHiddenLayer; j++) {
			fread(&bnh_variance[0][j], sizeof(float), 1, fbin);
			printf("%5.2f ", bnh_variance[0][j]);
		}
		printf("\n");

	// read other hidden layers
	for(i = 0; i < (nHiddenLayers - 1); i++) {
		// read layer (i+2) weights
		printf("weights_l%d\n", (i+2));
		for(j = 0; j < nNeuronsPerHiddenLayer; j++) {
			for(k = 0; k < nNeuronsPerHiddenLayer; k++) {
				fread(&weights_lh[i][j][k], sizeof(float), 1, fbin);
				printf("%5.2f ", weights_lh[i][j][k]);
			}
			printf("\n");
		}
		printf("bias_l%d\n", (i+2));
		for(j = 0; j < nNeuronsPerHiddenLayer; j++) {
				fread(&bias_lh[i][j], sizeof(float), 1, fbin);
				printf("%5.2f ", bias_lh[i][j]);
		}
		printf("\n");
		printf("Batch normalization parameters for hidden layer %d (+1):\n", i+1);
		for(j = 0; j < nNeuronsPerHiddenLayer; j++) {
			fread(&bnh_beta[i+1][j], sizeof(float), 1, fbin);
			printf("%5.2f ", bnh_beta[i+1][j]);
		}
		printf("\n");
		for(j = 0; j < nNeuronsPerHiddenLayer; j++) {
			fread(&bnh_gamma[i+1][j], sizeof(float), 1, fbin);
			printf("%5.2f ", bnh_gamma[i+1][j]);
		}
		printf("\n");
		for(j = 0; j < nNeuronsPerHiddenLayer; j++) {
			fread(&bnh_mean[i+1][j], sizeof(float), 1, fbin);
			printf("%5.2f ", bnh_mean[i+1][j]);
		}
		printf("\n");
		for(j = 0; j < nNeuronsPerHiddenLayer; j++) {
			fread(&bnh_variance[i+1][j], sizeof(float), 1, fbin);
			printf("%5.2f ", bnh_variance[i+1][j]);
		}
		printf("\n");
	}

	// read layer o weights
	printf("weights_lo\n");
	for(i = 0; i < nNeuronsPerHiddenLayer; i++) {
		for(j = 0; j < nOutputs; j++) {
			fread(&weights_lo[i][j], sizeof(float), 1, fbin);
			printf("%5.2f ", weights_lo[i][j]);
		}
		printf("\n");
	}
	// read bias_lo
	printf("bias_lo\n");
	for(j = 0; j < nOutputs; j++) {
		fread(&bias_lo[j], sizeof(float), 1, fbin);
		printf("%5.2f ", bias_lo[j]);
	}
	printf("\n");
	// read batch normalization for output layer
	printf("Batch normalization parameters for output layer:\n");
	for(j = 0; j < nOutputs; j++) {
		fread(&bno_beta[j], sizeof(float), 1, fbin);
		printf("%5.2f ", bno_beta[j]);
	}
	printf("\n");
	for(j = 0; j < nOutputs; j++) {
		fread(&bno_gamma[j], sizeof(float), 1, fbin);
		printf("%5.2f ", bno_gamma[j]);
	}
	printf("\n");
	for(j = 0; j < nOutputs; j++) {
		fread(&bno_mean[j], sizeof(float), 1, fbin);
		printf("%5.2f ", bno_mean[j]);
	}
	printf("\n");
	for(j = 0; j < nOutputs; j++) {
		fread(&bno_variance[j], sizeof(float), 1, fbin);
		printf("%5.2f ", bno_variance[j]);
	}
	printf("\n\n");

#ifdef PYTHON_BEST_VALUES_INCLUDED
	// Read the best values obtained in python, in order to compare them with our values	
	float x;
	printf("python best test values\n");
	for(j = 0; j < TEST_SAMPLES_NR; j++) {
		for(i = 0; i < nOutputs; i++) {
			fread(&x, sizeof(float), 1, fbin);
			printf("%5.2f ", x);
			python_best_test[j][i] = x;
		}
		printf("\n");
	}
	printf("\n");
	printf("python best idx test values\n");
	for(i = 0; i < TEST_SAMPLES_NR; i++) {
		fread(&j, sizeof(int), 1, fbin);
		python_best_idx_test[i] = j;
		printf("%d ", j);
		// python writes an extra int
		fread(&j, sizeof(int), 1, fbin);
	}
	printf("\n");
#endif
	fclose(fbin);
}

float hard_sigmoid(float x) {
	float aux;
	aux = (x + 1.)/2.;
	if(aux < 0)
		aux = 0;
	if(aux > 1)
		aux = 1;
	return aux;
}

float round_through(float x) {
    //Element-wise rounding to the closest integer with full gradient propagation.
    //A trick from [Sergey Ioffe](http://stackoverflow.com/a/36480182)
    //a op that behave as f(x) in forward mode,
    //but as g(x) in the backward mode.
	float rounded;
	if(x == 0.5)
		x = x - 0.001;
	else if(x == -0.5)
		x = x + 0.001;
	rounded = roundf(x);
	//return x + tf.stop_gradient(rounded - x);
	return rounded;
}

float binary_tanh_unit(float x) {
    return 2.*round_through(hard_sigmoid(x))-1.;
}

float fact_hidden_binary(float x) 
{
	// binary tanh is sgn
	return (x >= 0) ? 1. : -1.;
}


float fact_hidden(float x)
{
	return fact_hidden_binary(x);
	//return binary_tanh_unit(x);
}

float fact_output(float x) 
{
	//return fact_hidden(x);
	return x;
	//return (x >= 0) ? 1. : -1.;
}

float bnh(int l, int k, float wx)
{
	float bn;
	//return (w - mean) * variance * gamma + beta
	bn = ((wx - bnh_mean[l][k]) * bnh_variance[l][k] * bnh_gamma[l][k] + bnh_beta[l][k]);
	return bn;
}

float bno(int k, float wx)
{
	float bn;
	//return (w - mean) * variance * gamma + beta
	bn = ((wx - bno_mean[k]) * bno_variance[k] * bno_gamma[k] + bno_beta[k]);
	return bn;
}

int inference(int samplesNr, float *data, float *correct_output)
{
	int i, j, k, l, n;
	float wx[BNN_NEURONS_PER_HIDDEN_LAYER_N], bn;
	float result[TRAINING_SAMPLES_NR][BNN_OUTPUTS_N];

	//for(i = 0; i < TRAINING_SAMPLES_NR; i++) {
	for(i = 0; i < samplesNr; i++) {
	//for(i = 0; i < 1; i++) {
		// hidden layer 1
		for(k = 0; k < nNeuronsPerHiddenLayer; k++) { // 1st hidden layer
			wx[k] = bias_l1[k];
			for(j = 0; j < nInputs; j++) // inputs
				//wx[k] += weights_l1[j][k] * training_data[i][j];
				wx[k] += weights_l1[j][k] * (*(data + i * nInputs + j));
			//printf("wx[%d]=%3.1f ", k, wx[k]);
			bn = bnh(0, k, wx[k]);
			//values_hidden[0][k] = fact_hidden(wx[k]);
			values_hidden[0][k] = fact_hidden(bn);
		}
		//printf("\n");
		// other hidden layers
		for(l = 1; l < nHiddenLayers; l++) 
			for(k = 0; k < nNeuronsPerHiddenLayer; k++) { // current layer
				wx[k] = bias_lh[l-1][k];
				for(j = 0; j < nNeuronsPerHiddenLayer; j++) // previous layer
					wx[k] += weights_lh[l-1][j][k] * values_hidden[l-1][j];
				bn = bnh(l, k, wx[k]);
				//values_hidden[l][k] = fact_hidden(wx[k]);
				values_hidden[l][k] = fact_hidden(bn);
			}
		// output layer
		for(k = 0; k < nOutputs; k++) { // output layer
			wx[k] = bias_lo[k];
			for(j = 0; j < nNeuronsPerHiddenLayer; j++) // previous layer
				wx[k] += weights_lo[j][k] * values_hidden[nHiddenLayers - 1][j];
			bn = bno(k, wx[k]);
			//values_output[k] = fact_output(wx[k]);
			values_output[k] = fact_output(bn);
			//printf("i=%3d, k=%3d, \t bnn_out[%d]=%2.1f \t training_outputs[%d][%d]=%1.0f\n", 
			//	i, k, k, values_output[k], i, k, training_outputs[i][k]);
			printf("i=%3d, k=%3d, \t bnn_out[%d]=%2.1f \t correct_output[%d][%d]=%1.0f\n", 
				i, k, k, values_output[k], i, k, *(correct_output + i*nOutputs + k));
			result[i][k] = values_output[k];
		}
		printf("\n");
//#define DEBUG_INFERENCE
#ifdef DEBUG_INFERENCE
		printf("values_hidden: \n");
		for(l = 0; l < nHiddenLayers; l++) {
			printf("hidden layer %d (+ 1):\n\t", l);
			for(j = 0; j < nNeuronsPerHiddenLayer; j++)
				printf("%2.1f ", values_hidden[l][j]);
			printf("\n");
		}
		printf("\n");
#endif
	}

	int idmax=0, nCorrects=0;	
	printf("Correct outputs:\n");
	for(i = 0; i < samplesNr; i++) {
		idmax = 0;
		for(k = 0; k < nOutputs; k++) {
			if(*(correct_output + i * nOutputs + k) > *(correct_output + i * nOutputs + idmax))
				idmax = k;
		}
		printf("%d ", idmax);
	}
	printf("\n");
	printf("Obtained outputs:\n");
	for(i = 0; i < samplesNr; i++) {
		idmax = 0;
		for(k = 0; k < nOutputs; k++) {
			if(result[i][k] > result[i][idmax])
				idmax = k;
		}
		printf("%d ", idmax);
		if(*(correct_output + i * nOutputs + idmax) > 0.5)
			nCorrects++;
	}
	printf("\n");
	printf("Accuracy: %.2f\n", 100 * ((float)nCorrects) / samplesNr);
	return 0;
}

float thresholdh(int l, int k, float wx)
{
	//float threshold;
	//threshold = mean - (beta / (gamma * variance));
	//threshold = bnh_mean[l][k] - (bnh_beta[l][k] / (bnh_gamma[l][k] * bnh_variance[l][k]));
	//if((bnh_gamma[l][k] * bnh_variance[l][k]) < 0)
	//	wx = -wx;
	if(wx >= th[l][k])
		return 1;
	else
		return -1;

}

float thresholdo(int k, float wx)
{
	//float threshold;
	//threshold = mean - (beta / (gamma * variance));
	//threshold = bno_mean[k] - (bno_beta[k] / (bno_gamma[k] * bno_variance[k]));
	//if((bno_gamma[k] * bno_variance[k]) < 0)
	//	wx = -wx;
	//if(wx >= to[x])
	//	return 1;
	//else
	//	return -1;
	return wx;
}

float thresholdh_binarized(int l, int k, float wx)
{
	if(l == 0) {
		if(wx > th[l][k])
			return 1;
		else
			return 0;
	} else {
		if(wx > (int)th[l][k])
			return 1;
		else 
			return 0;
	}
}

float thresholdo_binarized(int k, float wx)
{
	//float threshold;
	//threshold = mean - (beta / (gamma * variance));
	//threshold = bno_mean[k] - (bno_beta[k] / (bno_gamma[k] * bno_variance[k]));
	//if((bno_gamma[k] * bno_variance[k]) < 0)
	//	wx = -wx;
	//if(wx > (int)to[k])
	//	return 1;
	//else
	//	return 0;
	return wx;

}

void compute_thresholds() {
	int i, j, k, l;

	printf("Thresholds:\n");
	// hidden layers
	for(l = 0; l < nHiddenLayers; l++) {
		for(k = 0; k < nNeuronsPerHiddenLayer; k++) {
			th[l][k] = bnh_mean[l][k] - (bnh_beta[l][k] / (bnh_gamma[l][k] * bnh_variance[l][k]));
			printf("%4.1f ", th[l][k]);
			if((bnh_gamma[l][k] * bnh_variance[l][k]) < 0) {
				// flip sign for W and bias
				if(l  == 0) {
					bias_l1[k] = -bias_l1[k];
					for(j = 0; j < nInputs; j++)
						weights_l1[j][k] = -weights_l1[j][k];
				} else {
					bias_lh[l][k] = -bias_lh[l][k];
					for(j = 0; j < nNeuronsPerHiddenLayer; j++)
						weights_lh[l][j][k] = -weights_lh[l][j][k];
				}
			}
		}
		printf("\n");
	}

	// output layer
	for(k = 0; k < nOutputs; k++) { // output layer
		to[k] = bno_mean[k] - (bno_beta[k] / (bno_gamma[k] * bno_variance[k]));
		printf("%4.1f ", to[k]);
		if((bno_gamma[k] * bno_variance[k]) < 0) {
			// flip sign for W and bias
			bias_lo[k] = -bias_lo[k];
			for(j = 0; j < nNeuronsPerHiddenLayer; j++) 
				weights_lo[j][k] = -weights_lo[j][k];
		}
	}
	printf("\n");
}

void binarization() {
	int i, j, k, l;

	// layer 1 weights
	for(i = 0; i < nInputs; i++) {
		for(j = 0; j < nNeuronsPerHiddenLayer; j++)
			if(weights_l1[i][j] < 0)
				weights_l1[i][j] = 0;
	}
	// other hidden layers weights
	for(i = 0; i < (nHiddenLayers - 1); i++) {
		for(j = 0; j < nNeuronsPerHiddenLayer; j++) {
			for(k = 0; k < nNeuronsPerHiddenLayer; k++)
				if(weights_lh[i][j][k] < 0)
					weights_lh[i][j][k] = 0;
		}
	}
	// layer o weights
	for(i = 0; i < nNeuronsPerHiddenLayer; i++) {
		for(j = 0; j < nOutputs; j++) {
			if(weights_lo[i][j] < 0)
				weights_lo[i][j] = 0;
		}
	}

	// Thresholds
	// threshold = (threshold + fanIn) / 2
	// hidden layers, except 1st layer
	for(l = 1; l < nHiddenLayers; l++) {
		for(k = 0; k < nNeuronsPerHiddenLayer; k++) {
			th[l][k] = (th[l][k] + nNeuronsPerHiddenLayer) / 2;
		}
	}
	// output layer
	for(k = 0; k < nOutputs; k++) { // output layer
		to[k] = (to[k] + nNeuronsPerHiddenLayer) / 2;
	}
}

void save_bnn() {
	int i, j, k, l;
	FILE *fout;

	printf("save_bnn:\n");
	printf("Opening file: \"%s\"\n", fpga_bnn_filename);
	if((fout = fopen(fpga_bnn_filename, "wt")) == NULL) {
		printf("Erorr opening \"%s\"", fpga_bnn_filename);
		exit(1);
	}

	// layer 1 weights
	// transposed and reverse order.
	for(j = 0; j < nNeuronsPerHiddenLayer; j++) {
		printf("Wt_l1(%d)<=\"", j);
		fprintf(fout, "Wt_l1(%d)<=\"", j);
		for(i = nInputs-1; i >=0; i--) {
			printf("%d", (int)weights_l1[i][j]);
			fprintf(fout, "%d", (int)weights_l1[i][j]);
		}
		printf("\";\n");
		fprintf(fout, "\";\n");
	}
	// other hidden layers weights
	for(i = 0; i < (nHiddenLayers - 1); i++) {
		// transposed and reverse order.
		for(k = 0; k < nNeuronsPerHiddenLayer; k++) {
			printf("Wt_lh%d(%d)<=\"", i+2, k);
			fprintf(fout, "Wt_lh%d(%d)<=\"", i+2, k);
			for(j = nNeuronsPerHiddenLayer-1; j >= 0; j--) {
				printf("%d", (int)weights_lh[i][j][k]);
				fprintf(fout, "%d", (int)weights_lh[i][j][k]);
			}
			printf("\";\n");
			fprintf(fout, "\";\n");
		}
	}
	// layer o weights
	for(j = 0; j < nOutputs; j++) {
		printf("Wt_lo(%d)<=\"", j);
		fprintf(fout, "Wt_lo(%d)<=\"", j);
		for(i = nNeuronsPerHiddenLayer-1; i>=0; i--) {
			printf("%d", (int)weights_lo[i][j]);
			fprintf(fout, "%d", (int)weights_lo[i][j]);
		}
		printf("\";\n");
		fprintf(fout, "\";\n");
	}

	// Thresholds
	printf("Thresholds:\n");
	//fprintf(fout, "Thresholds:\n");
	// Print layer 1 thresholds as float.
	//printf("Thresholds l1 float:\n");
	//fprintf(fout, "Thresholds l1 float:\n");
	for(k = 0; k < nNeuronsPerHiddenLayer; k++) {
		printf("%f ", th[0][k]);
		//fprintf(fout, "%f ", th[0][k]);
	}
	printf("\n");
	fprintf(fout, "\n");	
	half hf;
	for(l = 0; l < nHiddenLayers; l++) {
		for(k = 0; k < nNeuronsPerHiddenLayer; k++) {
			if(l == 0) {
				// Print layer 1 thresholds as half float short int.
				hf = th[l][k];
				//printf("thresholds_l1(%d) <= x\"%4x\"; ", k, *(unsigned short int*)&hf);
				fprintf(fout, "thresholds_l1(%d) <= x\"%4x\"; ", k, *(unsigned short int*)&hf);
			} else {
				printf("%d ", (int)th[l][k]);
				// some thresholds are greater than the number of neurons per layer.
				fprintf(fout, "thresholds_lh%d(%d) <= conv_std_logic_vector(%d,%d); ", 
					l+1, k, (int)th[l][k], (int)ceil(log2(BNN_NEURONS_PER_HIDDEN_LAYER_N))+1);
			}
		}
		printf("\n");
		fprintf(fout, "\n");
	}
	// output layer
	for(k = 0; k < nOutputs; k++) {
		printf("%d  ", (int)to[k]);
		// some thresholds are greater than the number of neurons per layer.
		fprintf(fout, "thresholds_lo(%d) <= conv_std_logic_vector(%d,%d); ", k, (int)to[k], (int)ceil(log2(BNN_NEURONS_PER_HIDDEN_LAYER_N))+1);
	}	
	printf("\n");
	fprintf(fout, "\n");

	//printf("BNN saved in %s \n", fpga_bnn_filename);
	fclose(fout);
}

int inference_threshold(int samplesNr, float *data, float *correct_output)
{
	int i, j, k, l, n;
	float wx[BNN_NEURONS_PER_HIDDEN_LAYER_N], bn;
	float result[TRAINING_SAMPLES_NR][BNN_OUTPUTS_N];

	//for(i = 0; i < TRAINING_SAMPLES_NR; i++) {
	for(i = 0; i < samplesNr; i++) {
	//for(i = 0; i < 1; i++) {
		// hidden layer 1
		for(k = 0; k < nNeuronsPerHiddenLayer; k++) { // 1st hidden layer
			//wx[k] = bias_l1[k]; 
			wx[k] = 0;
			for(j = 0; j < nInputs; j++) // inputs
				//wx[k] += weights_l1[j][k] * training_data[i][j];
				//wx[k] += weights_l1[j][k] * (*(data + i * nInputs + j));
				if(weights_l1[j][k] > 0)
					wx[k] += *(data + i * nInputs + j);
				else
					wx[k] -= *(data + i * nInputs + j);
			//printf("wx[%d]=%3.1f ", k, wx[k]);
			values_hidden[0][k] = thresholdh_binarized(0, k, wx[k]);
		}
		//printf("\n");
		// other hidden layers
		for(l = 1; l < nHiddenLayers; l++) 
			for(k = 0; k < nNeuronsPerHiddenLayer; k++) { // current layer
				wx[k] = 0;
				//wx[k] = bias_lh[l-1][k];
				for(j = 0; j < nNeuronsPerHiddenLayer; j++) // previous layer
					//wx[k] += weights_lh[l-1][j][k] * values_hidden[l-1][j];
					//if(weights_lh[l-1][j][k] == values_hidden[l-1][j])
					//	wx[k] += 1;
					wx[k] += !((int)weights_lh[l-1][j][k] ^ (int)values_hidden[l-1][j]);
				values_hidden[l][k] = thresholdh_binarized(l, k, wx[k]);
			}
		// output layer
		for(k = 0; k < nOutputs; k++) { // output layer
			wx[k] = 0;
			//wx[k] = bias_lo[k];
			for(j = 0; j < nNeuronsPerHiddenLayer; j++) // previous layer
				//wx[k] += weights_lo[j][k] * values_hidden[nHiddenLayers - 1][j];
				//if(weights_lo[j][k] == values_hidden[nHiddenLayers-1][j])
				//	wx[k] += 1;
				wx[k] += !((int)weights_lo[j][k] ^ (int)values_hidden[nHiddenLayers-1][j]);
			values_output[k] = thresholdo_binarized(k, wx[k]);
			printf("i=%3d, k=%3d, \t bnn_out[%d]=%2.1f \t correct_output[%d][%d]=%1.0f\n", 
				i, k, k, values_output[k], i, k, *(correct_output + i*nOutputs + k));
			result[i][k] = values_output[k];
		}
		printf("\n");

//#define DEBUG_INFERENCE
#ifdef DEBUG_INFERENCE
		printf("values_hidden: \n");
		for(l = 0; l < nHiddenLayers; l++) {
			printf("hidden layer %d (+ 1):\n\t", l);
			for(j = 0; j < nNeuronsPerHiddenLayer; j++)
				printf("%2.1f ", values_hidden[l][j]);
			printf("\n");
		}
		printf("\n");
#endif
	}

	int idmax=0, nCorrects=0;	
	printf("Correct outputs:\n");
	for(i = 0; i < samplesNr; i++) {
		idmax = 0;
		for(k = 0; k < nOutputs; k++) {
			if(*(correct_output + i * nOutputs + k) > *(correct_output + i * nOutputs + idmax))
				idmax = k;
		}
		printf("%d ", idmax);
	}
	printf("\n");
	printf("Obtained outputs:\n");
	for(i = 0; i < samplesNr; i++) {
		idmax = 0;
		for(k = 0; k < nOutputs; k++) {
			if(result[i][k] > result[i][idmax])
				idmax = k;
		}
		printf("%d ", idmax);
		if(*(correct_output + i * nOutputs + idmax) > 0.5)
			nCorrects++;
	}
	printf("\n");
	printf("Accuracy: %.2f\n", 100 * ((float)nCorrects) / samplesNr);
	return 0;
}	

void simple_experiments() {
	int i, j, k, l;
	//printf("roundf(-0.8)=%f, roundf(-0.5)=%f, roundf(-0.4)=%f, roundf(0.5)=%f roundf(0.1)=%f roundf(0.6)=%f\n",
	//	roundf(-0.8), roundf(-0.5), roundf(-0.4), roundf(0.5), roundf(0.1), roundf(0.6));
  	//printf("Hello from TensorFlow C library version %s\n", TF_Version());
	//return 0;

	//float x=-1.0;
	//printf("%d=%x \n", *(int*)&x, *(int*)&x);
	//return 0;

	//half a(3.4), b(0.5);
	//half c = a + b;
	//c += 3;
	//std::cout << a << " " << b << " " << c << std::endl;
	//printf("a=%x b=%x c=%x \n", *(short int*)&a, *(short int*)&b, *(short int*)&c);

	//unsigned short int si=0x4cff;
	//a = *(half*)&si;
        //std::cout << a << std::endl;

	printf("Test data [TEST_SAMPLES_NR-1]\n");
	half hf;
	for(j = 0; j < nInputs; j++) {
		hf = test_data[2][j];
		printf("%x ", *(unsigned short int*)&hf);
	}
	printf("\n");
}

int main(int argc, char *argv[])
{
	//simple_experiments();
	//return 0;

	//readNNFromTextFile();
	readNNFromBinaryFile();
	load_iris_dataset();
	//printf("Values for training data:\n");
	//inference(TRAINING_SAMPLES_NR, (float*)training_data, (float*)training_outputs);
	printf("Values for test data:\n");
	//inference(TEST_SAMPLES_NR, (float*)test_data, (float*)test_outputs);
	compute_thresholds();
	binarization();
	save_bnn();
	inference_threshold(TEST_SAMPLES_NR, (float*)test_data, (float*)test_outputs);

	//simple_experiments();
	return 0;
}
