package com.afconsult;

import com.afconsult.examples.xor.XorInputLayer;
import com.afconsult.examples.xor.XorOutputLayer;
import com.afconsult.mind.HiddenLayer;
import com.afconsult.mind.Mind;

import java.util.ArrayList;
import java.util.List;

public class Main {

    public static void main(String[] args) {
        XorInputLayer inputLayer = new XorInputLayer();

        List<HiddenLayer> hiddenLayers = new ArrayList<>(1);
        for (int i = 0; i < 5; i++) {
            hiddenLayers.add(new HiddenLayer(3, Mind.SIGMOID));
        }

        XorOutputLayer outputLayer = new XorOutputLayer();

        Mind mind = new Mind(inputLayer, hiddenLayers, outputLayer);

        mind.train();
    }
}
