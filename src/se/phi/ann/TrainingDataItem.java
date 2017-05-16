package se.phi.ann;

import se.phi.math.Matrix;

public abstract class TrainingDataItem {

    public abstract Matrix getInputData();

    public abstract Matrix getReferenceData();
}
