package se.phi.ann.xor;

import se.phi.ann.TrainingData;
import se.phi.ann.TrainingDataItem;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class XorTrainingData implements TrainingData {

    private List<TrainingDataItem> inputDataItems = Collections.unmodifiableList(
            Arrays.asList(
                    new XorTrainingDataItem(0, 0, new int[] {0}),
                    new XorTrainingDataItem(0, 1, new int[] {1}),
                    new XorTrainingDataItem(1, 0, new int[] {1}),
                    new XorTrainingDataItem(1, 1, new int[] {0})
            ));

    private int idx = 0;
    private XorTrainingData() {}

    public static XorTrainingData getInstance() {
        return new XorTrainingData();
    }

    @Override
    public int getNbrInputs() {
        return 2;
    }

    @Override
    public int getNbrOutputs() {
        return 1;
    }

    @Override
    public TrainingDataItem nextItem() {
        if (!hasMore()) {
            idx = 0;
        }

        return inputDataItems.get(idx++);
    }

    @Override
    public boolean hasMore() {
        return idx < inputDataItems.size();
    }

    @Override
    public void reset() {
        idx = 0;
    }
}
