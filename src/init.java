import ann.classifier.MatrixExceptionHandler;
import ann.classifier.NeuralNetwork;
import ann.classifier.Dataset;

public class init {
    public static void main(String[] args) throws MatrixExceptionHandler {
        NeuralNetwork ann = new NeuralNetwork(2, 64, 1, 3, "adam", 10000);
        Dataset x = new Dataset();
        ann.setLogs(true);
        ann.train(x.getSpiral_data(),x.getSpiral_classes());
    }
}

