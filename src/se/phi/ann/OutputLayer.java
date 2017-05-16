package se.phi.ann;

import se.phi.math.Matrix;

import java.util.function.Function;

public class OutputLayer extends Layer {


    OutputLayer(int thickness, Function<Double, Double> activationFunction) {
        super(thickness, activationFunction);

        terminal = true;
    }

    @Override
    public void forward(Matrix values) {
        input = values;
        output = values.multiply(weights).apply(activationFunction);
    }
}
