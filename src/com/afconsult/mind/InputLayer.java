package com.afconsult.mind;

import com.afconsult.math.Matrix;

public interface InputLayer extends Layer {

    void forward(Matrix inputValues);
}
