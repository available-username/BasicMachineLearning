package se.phi.ann.examples.xor;

import se.phi.ann.TrainingData;
import se.phi.ann.TrainingDataItem;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class XorTrainingData implements TrainingData {

    private TrainingDataItem[] items = new TrainingDataItem[]
            {
                    new XorTrainingDataItem(0, 0, new int[]{0}),
                    new XorTrainingDataItem(0, 1, new int[]{1}),
                    new XorTrainingDataItem(1, 0, new int[]{1}),
                    new XorTrainingDataItem(1, 1, new int[]{0})
            };

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
    public TrainingDataItem[] getTrainingData() {
        return items;
    }
}
