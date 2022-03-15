import ann.classifier.*;
import python.extender.PythonInterpreter;
import java.io.IOException;


public class init {
    public static void main(String[] args) throws MatrixExceptionHandler, IOException {
        NeuralNetwork ann = new NeuralNetwork(2, 64, 1, 3, "sgd", 2300);
        Dataset x = new Dataset();
        Dataset y = new Dataset();
        ann.setLogs(true);
        ann.setMetrices(false);
        ann.set_l2_regularizer(5e-4, 5e-4);
        //ann.fit(x.get_training_spiral_data(), x.get_training_spiral_classes());
        //ann.predict(new Matrix(1,2,new double[][]{{0.0978738, 0.3729515}}), new Matrix(1,3,new double[][]{{1.0,2.0,0.0}}).transpose());

    }
}
