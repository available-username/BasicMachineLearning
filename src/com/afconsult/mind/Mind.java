package com.afconsult.mind;

import com.afconsult.math.Matrix;

import java.util.List;
import java.util.function.Function;

public class Mind {

    public static final Function<Double, Double> SIGMOID = (x) -> 1 / (1 + Math.exp(-x));
    public static final Function<Double, Double> HYPTAN  = (x) -> { double x2 = 2 * x; return (1 - Math.exp(-x2)) / (1 + Math.exp(x2)); };
    private OutputLayer inputLayer;
    private List<HiddenLayer> hiddenLayers;
    private InputLayer outputLayer;

    public Mind(OutputLayer inputLayer, List<HiddenLayer> hiddenLayers, InputLayer outputLayer) {
        this.inputLayer = inputLayer;
        this.hiddenLayers = hiddenLayers;
        this.outputLayer = outputLayer;


        OutputLayer predecessor = inputLayer;

        int nHiddenLayers = hiddenLayers.size();
        for (int i = 0; i < nHiddenLayers - 1; i++) {
            HiddenLayer hiddenLayer = hiddenLayers.get(i);
            hiddenLayer.setPredecessor(predecessor);
            hiddenLayer.setSuccessor(hiddenLayers.get(i + 1));

            predecessor = hiddenLayer;
        }

        HiddenLayer lastHiddenLayer = hiddenLayers.get(nHiddenLayers - 1);
        lastHiddenLayer.setPredecessor(predecessor);
        lastHiddenLayer.setSuccessor(outputLayer);
    }

    public void train() {
        forward();
    }

    private void forward() {
        Matrix input = inputLayer.getOutput();
        hiddenLayers.get(0).forward(input);
    }
}
