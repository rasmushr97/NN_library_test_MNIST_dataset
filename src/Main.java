import neuralnet.NeuralNetwork;
import neuralnet.Relu;
import neuralnet.Sigmoid;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

public class Main {

    public static void main(String[] args) {
        NeuralNetwork nn = new NeuralNetwork(784, 65, 10);
        NeuralNetwork.learningRate = 0.05;

        String trainCSV = "src/data/train.csv";

        Path trainPath = Paths.get(trainCSV);

        try (BufferedReader trainBr = Files.newBufferedReader(trainPath)) {
            trainBr.readLine();

            List<List<List<Double>>> trainingData = prepareData(trainBr);
            nn.split(trainingData.get(0), trainingData.get(1));

        } catch (IOException e) {
            e.printStackTrace();
        }

        nn.setSoftmaxIsEnabled(true);
        nn.setActivationFunction(new Relu());

        nn.train();
        double testResult = nn.test();

        System.out.println((testResult * 100) + "% Correct\n");
    }

    private static List<List<List<Double>>> prepareData(BufferedReader br) throws IOException {
        List<List<List<Double>>> res = new ArrayList<>();
        List<List<Double>> inputs = new ArrayList<>();
        List<List<Double>> targets = new ArrayList<>();

        String line;
        while ((line = br.readLine()) != null) {
            List<String> data = new ArrayList<>(Arrays.asList(line.split(",")));
            List<Double> dataFormatted = new ArrayList<>();

            String label = data.remove(0);

            for (String s : data) {
                int value = Integer.parseInt(s);
                Double d = (double) value / 255;
                dataFormatted.add(d);
            }

            List<Double> labelFormatted = new ArrayList<>();
            for (int i = 0; i < 10; i++) {
                if (i == Integer.parseInt(label)) {
                    labelFormatted.add(1.0);
                } else {
                    labelFormatted.add(0.0);
                }
            }

            inputs.add(dataFormatted);
            targets.add(labelFormatted);
        }

        res.add(inputs);
        res.add(targets);

        return res;
    }

}
