package se.phi.ann;

public interface TrainingData {

    int getNbrInputs();

    int getNbrOutputs();

    TrainingDataItem[] getTrainingData();
}
