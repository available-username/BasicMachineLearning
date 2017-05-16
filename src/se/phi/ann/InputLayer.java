package se.phi.ann;

import se.phi.math.Matrix;

public class InputLayer extends Layer {

    public InputLayer(int thickness) {
        super(thickness, (x) -> x);

        terminal = true;
    }

    @Override
    public void forward(Matrix values) {
        output = values;
        successor.forward(values);
    }
}
