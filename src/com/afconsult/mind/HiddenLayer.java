package com.afconsult.mind;

import com.afconsult.math.Matrix;

import java.util.function.Function;

public class HiddenLayer implements InputLayer, OutputLayer {

    private Function<Double, Double> activiationFunction;
    private OutputLayer predecessor;
    private InputLayer successor;
    private int thickness;

    private Matrix weights;
    private Matrix weightedOutputs;
    private Matrix outputs;

    public HiddenLayer(int thickness, Function<Double, Double> activationFunction) {
        this.thickness = thickness;
        this.activiationFunction = activationFunction;
    }

    void setPredecessor(OutputLayer predecessor) {
        this.predecessor = predecessor;

        weights = new Matrix(predecessor.getThickness(), thickness, (r, c) -> Math.random());
    }

    void setSuccessor(InputLayer successor) {
        this.successor = successor;
    }

    @Override
    public int getThickness() {
        return thickness;
    }

    @Override
    public void forward(Matrix input) {
        System.out.println(input);
        weightedOutputs = input.multiply(weights);
        outputs = weightedOutputs.apply(activiationFunction);
        //System.out.println(outputs);
        successor.forward(outputs);
    }

    @Override
    public Matrix getOutput() {
        return outputs;
    }
}
