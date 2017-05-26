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

    Layer(int thickness, boolean useBias, Function<Double, Double> activationFunction) {
        this.thickness = thickness;
        this.activationFunction = activationFunction;
        this.useBias = useBias;
    }

    /* Only used when creating an input layer wich is essentially a
     *  pass through layer.
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

    int getThickness() {
        return thickness;
    }

    void setSuccessor(Layer successor) {
        this.successor = successor;
    }

    Optional<Layer> getSuccessor() {
        return Optional.ofNullable(successor);
    }

    private double getInitialWeight() {
        return Math.random() - 0.5;
    }

    void setPredecessor(Layer predecessor) {
        this.predecessor = predecessor;
    }

    void generateWeights() {
        weights = new Matrix(predecessor.getThickness(), getThickness(), (r, c) -> getInitialWeight());

        if (useBias) {
            biasWeights = new Matrix(1, thickness, (r, c) -> getInitialWeight());
        }
    }

    Matrix feedForward(Matrix input) {
        this.input = input;

        if (predecessor != null) {
            output = (useBias ? input.multiply(weights).add(biasWeights) : input.multiply(weights)).apply(activationFunction);
        } else {
            output = input;
        }

        return successor != null ? successor.feedForward(output) : output;
    }

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

    Matrix getOutput() {
        return output;
    }

    Matrix getWeights() {
        return weights;
    }

    Matrix getBiasWeights() {
        if (!useBias) {
            throw new IllegalStateException("Layer was not created with active bias");
        }
        return biasWeights;
    }

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
