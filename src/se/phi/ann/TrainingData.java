package se.phi.ann;

public interface TrainingData {

    int getNbrInputs();

    int getNbrOutputs();

    TrainingDataItem nextItem();

    boolean hasMore();

    void reset();
}
