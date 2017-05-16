package se.phi.ann;

import junit.framework.TestCase;
import se.phi.ann.xor.XorTrainingData;
import se.phi.math.Matrix;

import java.util.function.Function;

public class TopologyTest extends TestCase {

    public void testXorPrediction() {
        TrainingData trainingData = XorTrainingData.getInstance();
        int thickness = 2;
        int nbrHiddenLayers = 1;
        double learningRate = 0.001;
        int iterations = 1000000;

        Topology topology = new Topology(trainingData.getNbrInputs(), trainingData.getNbrOutputs(), nbrHiddenLayers, thickness, Topology.SIGMOID);
        topology.train(trainingData, learningRate, iterations);

        Matrix zeroZero = new Matrix(1, 2, (r, c) -> (double)0);
        Matrix zeroOne = new Matrix(1, 2, (r, c) -> (double)(c == 0 ? 0 : 1));
        Matrix oneZero = new Matrix(1, 2, (r, c) -> (double)(c == 0 ? 1 : 0));
        Matrix oneOne = new Matrix(1, 2, (r, c) -> (double)1);

        Function<Double, Double> toExtremes = (a) -> a > 0.5 ? 1.0 : 0.0;
        assertEquals(Matrix.Zeros(1, 1), topology.predict(zeroZero).apply(toExtremes));
        assertEquals(Matrix.Ones(1, 1), topology.predict(zeroOne).apply(toExtremes));
        assertEquals(Matrix.Ones(1, 1), topology.predict(oneZero).apply(toExtremes));
        assertEquals(Matrix.Zeros(1, 1), topology.predict(oneOne).apply(toExtremes));
    }
}