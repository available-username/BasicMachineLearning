package se.phi.ann;

import se.phi.math.Matrix;

import java.io.OutputStream;
import java.io.PrintStream;
import java.util.Optional;
import java.util.Scanner;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Function;

class Layer {

    private final int thickness;
    private Function<Double, Double> activationFunction;

    private Layer predecessor;
    private Layer successor;

    private Matrix input;
    private Matrix weights;
    private Matrix biasWeights;
    private Matrix output;
    private boolean useBias;

    /**
     * Represents a layer in the network. A network consists of an input layer,
     * an output layer and any number of hidden layers. A layer is classified according
     * to this matrix
     *
     *             | Has successor | Has predecessor
     *             +---------------+----------------
     * Input layer | Yes           | No
     *             +---------------+----------------
     * Hidden layer| Yes           | Yes
     *             +---------------+----------------
     * Output layer| No            | Yes
     *             +---------------+----------------
     *
     * @param thickness number of neurons in this layer, must be larger than 0
     * @param useBias {@code true} if a bias term should be used in the output calculation
     * @param activationFunction a sigmoid function
     */
    Layer(int thickness, boolean useBias, Function<Double, Double> activationFunction) {
        this.thickness = thickness;
        this.activationFunction = activationFunction;
        this.useBias = useBias;
    }

    /**
     *  Creates an input layer which is essentially a pass through layer.
     */
    Layer(int thickness) {
        this(thickness, false, null);
    }

    private Layer(double[][] weights, double[][] biasWeights) {
        this.activationFunction = Network.SIGMOID;
        this.weights = new Matrix(weights);
        this.thickness = this.weights.rows;

        useBias = biasWeights != null;

        if (useBias) {
            this.biasWeights = new Matrix(biasWeights);
        }
    }

    /**
     * Get the number of neurons in this layer.
     * @return number of neurons
     */
    int getThickness() {
        return thickness;
    }

    /**
     * Set the successor layer to this layer. Call {@link #generateWeights() generateWeights}
     * after {@link #setSuccessor(Layer) setSuccessor}.
     *
     * @param successor a layer
     */
    void setSuccessor(Layer successor) {
        this.successor = successor;
    }

    /**
     * Get the successor of this layer if any
     * @return an optional successor
     */
    Optional<Layer> getSuccessor() {
        return Optional.ofNullable(successor);
    }

    private double getInitialWeight() {
        return Math.random() - 0.5;
    }

    /**
     * Set the predecessor layer to this layer.
     * @param predecessor a layer
     */
    void setPredecessor(Layer predecessor) {
        this.predecessor = predecessor;
    }

    /**
     * Generate new weights for the biases if any and inputs received from the predecessor layer.
     */
    void generateWeights() {
        weights = new Matrix(predecessor.getThickness(), getThickness(), (r, c) -> getInitialWeight());

        if (useBias) {
            biasWeights = new Matrix(1, thickness, (r, c) -> getInitialWeight());
        }
    }

    /**
     * Feed data to the layer, calculate output and pass it on to the successor layer if any.
     * @param input a vector (1-by-N matrix) of input data
     * @return the resulting output vector (1-by-M matrix) from the last layer in the network
     */
    Matrix feedForward(Matrix input) {
        this.input = input;

        if (predecessor != null) {
            output = (useBias ? input.multiply(weights).add(biasWeights) : input.multiply(weights)).apply(activationFunction);
        } else {
            output = input;
        }

        return successor != null ? successor.feedForward(output) : output;
    }

    /**
     * Update weights and biases based to the error calculated by the successor, back propagate
     * the error calculated by this layer to it's predecessor.
     * @param error an error
     * @param learningRate a measure of how dramatically the weights should be updated. Suitable values
     *                     are in the range (0,1]
     */
    void backPropagate(Matrix error, double learningRate) {
        if (predecessor != null) {
            Matrix derivative = Matrix.diagonalize(output.apply(x -> x * (1 - x)));

            Matrix backPropagatedError = derivative.multiply(error);

            Matrix deltaWeights = backPropagatedError.multiply(input).scale(learningRate).transpose();
            weights = weights.subtract(deltaWeights);

            if (useBias) {
                Matrix deltaBiasWeights = backPropagatedError.scale(learningRate).transpose();
                biasWeights = biasWeights.subtract(deltaBiasWeights);
            }

            predecessor.backPropagate(weights.multiply(backPropagatedError), learningRate);
        }
    }

    /**
     * Get the output produced by this layer.
     * @return a vector (1-by-M) matrix
     */
    Matrix getOutput() {
        return output;
    }

    /**
     * Get the weights of this layer.
     * @return an N-by-M matrix
     */
    Matrix getWeights() {
        return weights;
    }

    /**
     * Get the bias weights of this layer.
     * @return an 1-by-M matrix or {@code null} if biases are not used
     */
    Matrix getBiasWeights() {
        if (!useBias) {
            throw new IllegalStateException("Layer was not created with active bias");
        }
        return biasWeights;
    }

    /**
     * Save layer.
     * @param outputStream
     */
    void save(OutputStream outputStream) {
        try (PrintStream printStream = new PrintStream(outputStream)) {
            int rows = weights.getRows();
            int cols = weights.getCols();

            printStream.format("%d %d\n", rows, cols);

            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    if (c == 0) {
                        printStream.print(weights.get(r, c));
                    } else {
                        printStream.format(" %f", weights.get(r, c));
                    }
                }
                printStream.println();
            }
        }
    }

    /**
     * Load layer.
     * @param scanner scanner attahed to an {@code InputStream}
     * @param useBias {@code true} if a bias matrix is present in the data
     * @return an layer instance
     */
    static Layer load(Scanner scanner, boolean useBias) {
        double[][] weights = loadMatrix(scanner);
        double[][] biasWeights = useBias ? loadMatrix(scanner) : null;

        return new Layer(weights, biasWeights);
    }

    private static double[][] loadMatrix(Scanner scanner) {
        int rows = scanner.nextInt();
        int cols = scanner.nextInt();

        double[][] m = new double[rows][];

        for (int r = 0; r < rows; r++) {
            m[r] = new  double[cols];

            for (int c = 0; c < cols; c++) {
                m[r][c] = scanner.nextDouble();
            }
        }

        return m;
    }
}
