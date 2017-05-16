package se.phi.ann;

import se.phi.math.Matrix;

import java.util.function.Function;

public class Topology {

    public static final Function<Double, Double> SIGMOID =
            (x) -> 1 / (1 + Math.exp(-x));

    public static final Function<Double, Double> HYPTAN  =
            (x) -> { double x2 = 2 * x; return (1 - Math.exp(-x2)) / (1 + Math.exp(x2)); };

    private static final int DEFAULT_NBR_TRAINING_ITERATIONS = 10000;
    private static final double DEFAULT_LEARNING_RATE = 0.01;

    private final InputLayer inputLayer;
    private final OutputLayer outputLayer;

    public Topology(int nbrInputNodes, int nbrOutputNodes, int nbrHiddenLayers, int hiddenLayerThickness, Function<Double, Double> activationFunction) {
        inputLayer = new InputLayer(nbrInputNodes);
        outputLayer = new OutputLayer(nbrOutputNodes, activationFunction);

        Layer predecessor = inputLayer;

        for (int i = 0; i < nbrHiddenLayers; i++) {
            Layer hiddenLayer = new Layer(hiddenLayerThickness, activationFunction);
            predecessor.setSuccessor(hiddenLayer);
            hiddenLayer.setPredecessor(predecessor);
            predecessor = hiddenLayer;
        }

        predecessor.setSuccessor(outputLayer);
        outputLayer.setPredecessor(predecessor);
    }

    public Matrix predict(Matrix input) {
        inputLayer.forward(input);
        return outputLayer.getOutput();
    }

    public void train(TrainingData trainingData) {
        train(trainingData, DEFAULT_LEARNING_RATE, DEFAULT_NBR_TRAINING_ITERATIONS);
    }

    public void train(TrainingData trainingData, double learningRate, int iterations) {
        if (iterations < 1) {
            throw new IllegalArgumentException("At least one iteration required");
        }

        for (int i = 0; i < iterations; i++) {
            while (trainingData.hasMore()) {
                TrainingDataItem trainingDataItem = trainingData.nextItem();
                Matrix input = trainingDataItem.getInputData();
                Matrix reference = trainingDataItem.getReferenceData();

                // Feed forward the input signal
                inputLayer.forward(input);

                // Read the prediction for the output layer
                Matrix output = outputLayer.getOutput();

                // Calculate the error between the prediction and the output signal
                Matrix error = calculateError(reference, output);

                Layer current = outputLayer;
                Layer predecessor = current.predecessor;

                // Calculate the change in the output layer weights but don't apply them until later
                // when all the hidden layer weights have been adjusted. The deltas are the outputs
                // from each neuron multiplied by the corresponding error and the learning rate
                Matrix o = predecessor.getOutput();
                Matrix finalError1 = error;
                Matrix outputLayerDeltaWeights = new Matrix(predecessor.getThickness(), current.getThickness(),
                        (r, c) -> o.get(0, r) * finalError1.get(0, c) * learningRate);

                while (!predecessor.isTerminal()) {
                    // Calculate error terms for the hidden nodes
                    Matrix weights = current.getWeights();
                    Matrix finalError = error;
                    Matrix wErrors = new Matrix(weights.rows, weights.cols,
                            (r, c) -> weights.get(r, c) * finalError.get(0, c)).transpose();

                    Matrix predecessorOutput = predecessor.getOutput();

                    Matrix partialDeltaWeight = new Matrix(wErrors.rows, wErrors.cols,
                            (r, c) -> wErrors.get(r, c) * predecessorOutput.get(0, c) * (1 - predecessorOutput.get(0, c)));

                    Matrix pi = predecessor.getInput();
                    Matrix deltaWeight = new Matrix(pi.cols, predecessor.getThickness(),
                            (r, c) -> pi.get(0, r) * partialDeltaWeight.get(0, c) * learningRate);

                    predecessor.updateWeights(deltaWeight);

                    current = predecessor;
                    predecessor = current.predecessor;
                    error = wErrors;
                }

                outputLayer.updateWeights(outputLayerDeltaWeights);
            }

            trainingData.reset();
        }
    }

    private Matrix calculateError(Matrix reference, Matrix prediction) {
        // R = Reference
        // P = Prediction
        //
        // error = P * (1 - P) * (R - P)
        // i.e. the derivative of the prediction times the difference
        return calculateDerivative(prediction).multiplyElementWise(reference.subtract(prediction));
    }

    private Matrix calculateDerivative(Matrix matrix) {
        // Derivative property of Sigmoid function S
        // der(S) = S(1 - S)
        return matrix.apply(x -> x * (1 - x));
    }
}
