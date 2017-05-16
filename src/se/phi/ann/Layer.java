package se.phi.ann;

import se.phi.math.Matrix;

import java.util.function.Function;

public class Layer {

    private final int thickness;
    Function<Double, Double> activationFunction;

    Layer predecessor;
    Layer successor;

    Matrix input;
    Matrix weights;
    Matrix output;

    protected boolean terminal;

    Layer(int thickness, Function<Double, Double> activationFunction) {
        this.thickness = thickness;
        this.activationFunction = activationFunction;
    }

    int getThickness() {
        return thickness;
    }

    void setSuccessor(Layer successor) {
        this.successor = successor;
    }

    void setPredecessor(Layer predecessor) {
        this.predecessor = predecessor;
        weights = new Matrix(predecessor.getThickness(), getThickness(), (r, c) -> Math.random() - 0.5);
    }

    public void forward(Matrix values) {
        input = values;
        output = values.multiply(weights).apply(activationFunction);
        successor.forward(output);
    }

    Matrix getInput() {
        return input;
    }

    Matrix getOutput() {
        return output;
    }

    void updateWeights(Matrix delta) {
        weights = weights.add(delta);
    }

    Matrix getWeights() {
        return weights;
    }

    boolean isTerminal() {
        return terminal;
    }
}
