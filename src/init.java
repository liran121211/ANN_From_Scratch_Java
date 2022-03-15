import ann.classifier.MatrixExceptionHandler;
import ann.classifier.NeuralNetwork;
import ann.classifier.Dataset;
import python.extender.PythonInterpreter;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;


public class init {
    public static void main(String[] args) throws MatrixExceptionHandler, IOException {
        NeuralNetwork ann = new NeuralNetwork(2, 64, 1, 3, "sgd", 2300);
        Dataset x = new Dataset();
        Dataset y = new Dataset();
        ann.setLogs(true);
        ann.setMetrices(true);
        ann.set_l2_regularizer(5e-4, 5e-4);
        ann.fit(x.get_training_spiral_data(), x.get_training_spiral_classes());
        ann.predict(y.get_training_spiral_data(), y.get_training_spiral_classes());
    }
}
