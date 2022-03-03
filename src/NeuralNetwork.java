import java.util.*;

public class NeuralNetwork {
    final int[] input_size;
    final int[] hidden_size;
    Matrix input_layer;
    Matrix hidden_layer;
    Matrix output_layer;

    protected NeuralNetwork(Matrix inputs, Matrix hidden) {
        this.input_size = new int[]{inputs.getRows(), inputs.getColumns()};
        this.hidden_size = new int[]{hidden.getRows(), hidden.getColumns()};

        this.input_layer = inputs;
        this.hidden_layer = hidden;

    }



    public static void main(String[] args) {
        Matrix inputs = new Matrix(3, 4, new double[][]{{1.0, 2.0, 3.0, 2.5}, {2.0, 5.0, -1.0, 2.0}, {-1.5, 2.7, 3.3, -0.8}});
        Matrix weights = new Matrix(3, 4, new double[][]{{0.2, 0.8, -0.5, 1.0}, {0.5, -0.91, 0.26, -0.5}, {-0.26, -0.27, 0.17, 0.87}});
        Matrix bias = new Matrix(1, 3, new double[]{2.0, 3.0, 0.5});

        Matrix weights2 = new Matrix(3,3,new double[][]{{0.1,-0.14,0.5},{-0.5,0.12,-0.33},{-0.44,0.73,-0.13}});
        Matrix bias2 = new Matrix(1, 3, new double[]{-1.0,2.0,-0.5});
        Matrix layer1_outputs = addBias(inputs.product(weights.transpose()),bias);
        Matrix layer2_outputs = addBias(layer1_outputs.product(weights2.transpose()),bias2);

        System.out.println(layer1_outputs);
        System.out.println(layer2_outputs);
        System.out.println(Matrix.random(1000,1000));
    }

    protected static Matrix addBias(Matrix B, Matrix V) {
        for (int i = 0; i < B.getRows(); i++) {
            for (int j = 0; j < B.getColumns(); j++)
                B.setValue(i, j, V.getValue(0, j) + B.getValue(i, j));
        }
        return B;
    }
}

