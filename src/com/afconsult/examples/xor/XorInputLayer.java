package com.afconsult.examples.xor;

import com.afconsult.math.Matrix;
import com.afconsult.mind.OutputLayer;

public class XorInputLayer implements OutputLayer {

    @Override
    public int getThickness() {
        return 2;
    }

    @Override
    public Matrix getOutput() {
        //return Matrix.Zeros(1, getThickness());
        return new Matrix(1, getThickness(), (r, c) -> 1.0);
    }
}
