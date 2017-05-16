package com.afconsult.examples.xor;

import com.afconsult.math.Matrix;
import com.afconsult.mind.InputLayer;

public class XorOutputLayer implements InputLayer {

    @Override
    public int getThickness() {
        return 1;
    }

    @Override
    public void forward(Matrix inputValues) {
        System.out.println(inputValues);
    }
}
