import ann.classifier.MatrixExceptionHandler;
import ann.classifier.NeuralNetwork;
import ann.classifier.Dataset;

import java.awt.*;

public class init {
    public static void main(String[] args) throws MatrixExceptionHandler {
        NeuralNetwork ann = new NeuralNetwork(2, 64, 1, 3, "adam", 10000);
        Dataset x = new Dataset();
        ann.setLogs(true);
        ann.set_l2_regularizer(5e-4,5e-4);
        ann.fit(x.getSpiral_data(),x.getSpiral_classes());

    }
}

