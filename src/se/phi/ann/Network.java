package se.phi.ann;

import se.phi.math.Matrix;

import java.io.InputStream;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.*;
import java.util.concurrent.ThreadLocalRandom;
import java.util.function.Function;

public class Network {

    public static final Function<Double, Double> SIGMOID =
            (x) -> 1.0 / (1.0 + Math.exp(-x));

    public static final Function<Double, Double> HYPTAN  =
            (x) -> { double x2 = 2 * x; return (1 - Math.exp(-x2)) / (1 + Math.exp(x2)); };

    private Layer inputLayer;
    private Layer outputLayer;
    private boolean useBias;

    public Network(int nbrInputs, List<Integer> topology, int nbrOutputs, boolean useBias, Function<Double, Double> activationFunction) {
        inputLayer = new Layer(nbrInputs, useBias, activationFunction);
        outputLayer = new Layer(nbrOutputs, useBias, activationFunction);
        this.useBias = useBias;
        Layer predecessor = inputLayer;

        for (int thickness : topology) {
            Layer layer = new Layer(thickness, useBias, activationFunction);
            layer.setPredecessor(predecessor);
            layer.generateWeights();
            predecessor.setSuccessor(layer);
            predecessor = layer;
        }

        outputLayer.setPredecessor(predecessor);
        outputLayer.generateWeights();
        predecessor.setSuccessor(outputLayer);
    }

    private Network(List<Layer> layers) {
        int nbrInputs = layers.get(0).getThickness();
        inputLayer = new Layer(nbrInputs);
        outputLayer = layers.remove(layers.size() - 1);

        Layer predecessor = inputLayer;

        for (Layer layer : layers) {
            layer.setPredecessor(predecessor);
            predecessor.setSuccessor(layer);
            predecessor = layer;
        }

        outputLayer.setPredecessor(predecessor);
        predecessor.setSuccessor(outputLayer);
    }

    public Matrix predict(Matrix input) {
        inputLayer.feedForward(input);
        return outputLayer.getOutput();
    }

    public double train(TrainingData trainingData, double learningRate, int nbrEpochs, int nbrBatches) {
        double quadError = 0;
        long iteration = 1;

        for (int epoch = 0; epoch < nbrEpochs; epoch++) {
            Collection<TrainingDataItem[]> miniBatches = getMiniBatches(trainingData.getTrainingData(), nbrBatches);

            for (TrainingDataItem[] batch : miniBatches) {
                for (TrainingDataItem item : batch) {
                    Matrix output = inputLayer.feedForward(item.getInputData());
                    Matrix reference = item.getReferenceData();
                    Matrix error = output.subtract(reference);
                    Matrix errorTranspose = error.transpose();

                    quadError += (error.multiply(errorTranspose).get(0,0) - quadError) / iteration;
                    iteration += 1;

                    outputLayer.backPropagate(errorTranspose, learningRate);
                }
            }
        }

        return quadError;
    }

    private Collection<TrainingDataItem[]> getMiniBatches(TrainingDataItem[] trainingData, int nbrOfBatches) {
        shuffle(trainingData);

        int batchSize = trainingData.length / nbrOfBatches;

        Collection<TrainingDataItem[]> batches = new ArrayList<>();
        for (int b = 0; b < nbrOfBatches; b++) {
            TrainingDataItem[] batch = Arrays.copyOfRange(trainingData, b * batchSize, (b + 1) * batchSize);
            batches.add(batch);
        }

        return batches;
    }

    private void shuffle(TrainingDataItem[] data) {
        Random random = ThreadLocalRandom.current();

        for (int i = data.length - 1; i > 0; i--) {
            int swapIndex = random.nextInt(i);

            TrainingDataItem tmp = data[i];
            data[i] = data[swapIndex];
            data[swapIndex] = tmp;
        }
    }

    private void getTopology(List<Matrix> topology, Layer layer) {
        topology.add(layer.getWeights());

        if (useBias) {
            topology.add(layer.getBiasWeights());
        }

        layer.getSuccessor().ifPresent(successor -> getTopology(topology, successor));
    }

    private List<Matrix> getTopology() {
        List<Matrix> topology = new ArrayList<>();
        inputLayer.getSuccessor().ifPresent(successor -> getTopology(topology, successor));
        return topology;
    }

    public void save(OutputStream outputStream) {
        try (PrintStream printStream = new PrintStream(outputStream)) {

            List<Matrix> topology = getTopology();
            printStream.println(useBias);
            printStream.println(useBias ? topology.size() / 2 : topology.size());

            Iterator<Matrix> iterator = topology.iterator();
            while (iterator.hasNext()) {
                Matrix weights = iterator.next();
                outputMatrix(weights, printStream);

                if (useBias) {
                    Matrix biasWeights = iterator.next();
                    outputMatrix(biasWeights, printStream);
                }
            }
        }
    }

    private void outputMatrix(Matrix matrix, PrintStream printStream) {
        int rows = matrix.getRows();
        int cols = matrix.getCols();

        printStream.format("%d %d\n", rows, cols);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                if (c == 0) {
                    printStream.print(matrix.get(r, c));
                } else {
                    printStream.format(" %f", matrix.get(r, c));
                }
            }
            printStream.println();
        }
    }

    public static Network load(InputStream inputStream) {
        try (Scanner scanner = new Scanner(inputStream)) {
            scanner.useLocale(Locale.ENGLISH);
            boolean useBias = scanner.nextBoolean();
            int nbrLayers = scanner.nextInt();
            List<Layer> layers = new ArrayList<>(nbrLayers);

            for (int i = 0; i < nbrLayers; i++) {
                Layer layer = Layer.load(scanner, useBias);
                layers.add(layer);
            }

            return new Network(layers);
        }
    }
}
