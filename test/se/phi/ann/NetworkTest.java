package se.phi.ann;

import junit.framework.TestCase;
import se.phi.ann.examples.and.AndTrainingData;
import se.phi.ann.examples.sine.SinTrainingData;
import se.phi.ann.examples.xor.XorTrainingData;
import se.phi.math.Matrix;

import java.io.*;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.function.Function;

public class NetworkTest extends TestCase {

    public void testXorLearning() {
        TrainingData data = XorTrainingData.getInstance();

        Network net = new Network(data.getNbrInputs(), Arrays.asList(2), data.getNbrOutputs(), true, Network.SIGMOID);

        double error = net.train(data, 0.7, 10000, 1);

        Matrix i00 = new Matrix(new double[][] {{0, 0}});
        Matrix i01 = new Matrix(new double[][] {{0, 1}});
        Matrix i10 = new Matrix(new double[][] {{1, 0}});
        Matrix i11 = new Matrix(new double[][] {{1, 1}});

        System.out.println("" + i00 + " => " + net.predict(i00)); //.apply(x -> (double)(x > 0.5 ? 1 : 0)));
        System.out.println("" + i01 + " => " + net.predict(i01)); //.apply(x -> (double)(x > 0.5 ? 1 : 0)));
        System.out.println("" + i10 + " => " + net.predict(i10)); //.apply(x -> (double)(x > 0.5 ? 1 : 0)));
        System.out.println("" + i11 + " => " + net.predict(i11)); //.apply(x -> (double)(x > 0.5 ? 1 : 0)));

        String home = System.getProperty("user.home");
        Path path = Paths.get(home, "network", "xor.top");
        File file = path.toFile();
        if (!file.exists()) {
            file.getParentFile().mkdirs();
        }

        try (OutputStream outputStream = new FileOutputStream(file)) {
            net.save(outputStream);
        } catch (IOException e) {
            e.printStackTrace();
        }

        Network net2 = null;
        try (InputStream inputStream = new FileInputStream(file)) {
            net2 = Network.load(inputStream);

        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("" + i00 + " => " + net2.predict(i00)); //.apply(x -> (double)(x > 0.5 ? 1 : 0)));
        System.out.println("" + i01 + " => " + net2.predict(i01)); //.apply(x -> (double)(x > 0.5 ? 1 : 0)));
        System.out.println("" + i10 + " => " + net2.predict(i10)); //.apply(x -> (double)(x > 0.5 ? 1 : 0)));
        System.out.println("" + i11 + " => " + net2.predict(i11)); //.apply(x -> (double)(x > 0.5 ? 1 : 0)));

        /*
        System.out.println("Error: " + error);
        net.getTopology().forEach(System.out::println);

        Matrix i00 = new Matrix(new double[][] {{0, 0}});
        Matrix i01 = new Matrix(new double[][] {{0, 1}});
        Matrix i10 = new Matrix(new double[][] {{1, 0}});
        Matrix i11 = new Matrix(new double[][] {{1, 1}});

        System.out.println("" + i00 + " => " + net.predict(i00)); //.apply(x -> (double)(x > 0.5 ? 1 : 0)));
        System.out.println("" + i01 + " => " + net.predict(i01)); //.apply(x -> (double)(x > 0.5 ? 1 : 0)));
        System.out.println("" + i10 + " => " + net.predict(i10)); //.apply(x -> (double)(x > 0.5 ? 1 : 0)));
        System.out.println("" + i11 + " => " + net.predict(i11)); //.apply(x -> (double)(x > 0.5 ? 1 : 0)));
        */

    }
}