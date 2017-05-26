package se.phi.ann.examples.xor;

import se.phi.math.Matrix;
import se.phi.ann.TrainingDataItem;

class XorTrainingDataItem extends TrainingDataItem {

    private Matrix inputData;
    private Matrix referenceData;

    XorTrainingDataItem(int a, int b, int[] result) {
       inputData = new Matrix(1, 2, (r, c) -> (double)(c == 0 ? a : b));
       referenceData = new Matrix(1, result.length, (r, c) -> (double)result[c]);
    }

    @Override
    public Matrix getInputData() {
        return inputData;
    }

    @Override
    public Matrix getReferenceData() {
        return referenceData;
    }
}
