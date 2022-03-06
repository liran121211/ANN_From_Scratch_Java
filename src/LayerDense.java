import com.github.sh0nk.matplotlib4j.NumpyUtils;
import com.github.sh0nk.matplotlib4j.Plot;
import com.github.sh0nk.matplotlib4j.PythonExecutionException;
import com.github.sh0nk.matplotlib4j.builder.ContourBuilder;
import org.apache.log4j.helpers.FormattingInfo;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class LayerDense {
    private Matrix weights;
    private Matrix biases;
    private Matrix output;

    protected LayerDense(int n_inputs, int n_neurons) {
        this.weights = Matrix.random(n_inputs, n_neurons).multiply(0.01);
        this.biases = new Matrix(1, n_neurons);
    }

    /**
     * Calculate output values from inputs, weights and biases
     *
     * @param inputs (Matrix object).
     */
    protected void forward(Matrix inputs) {
        this.output = addBias(inputs.dot(this.weights), this.biases);
    }


    protected static Matrix addBias(Matrix B, Matrix V) throws IndexOutOfBoundsException {
        if (B.getColumns() != V.getColumns())
            throw new IndexOutOfBoundsException(String.format("Matrices has different dimensions (%s,%s) By (%s,%s)", B.getRows(), B.getColumns(), V.getRows(), V.getColumns()));

        for (int i = 0; i < B.getRows(); i++) {
            for (int j = 0; j < B.getColumns(); j++)
                B.setValue(i, j, V.getValue(0, j) + B.getValue(i, j));
        }
        return B;
    }

    public static void main(String[] args) throws PythonExecutionException, IOException {
        Matrix X = new Dataset().getVertical_data();
        Matrix y = new Dataset().getVertical_classes();
        LayerDense dense1 = new LayerDense(2, 3);
        LayerDense dense2 = new LayerDense(3, 3);

        Activation activation1 = new Activation_ReLU();
        Activation activation2 = new Activation_SoftMax();

        Loss_CategoricalCrossentropy loss_function = new Loss_CategoricalCrossentropy();


        double lowest_loss = 9999999;
        Matrix best_dense1_weights = new Matrix(dense1.weights);
        Matrix best_dense1_biases = new Matrix(dense1.biases);
        Matrix best_dense2_weights = new Matrix(dense2.weights);
        Matrix best_dense2_biases = new Matrix(dense2.biases);

        for (int i = 0; i < 10000; i++) {
            dense1.weights = dense1.weights.add(Matrix.random(2, 3).multiply(0.05));
            dense1.biases = dense1.biases.add(Matrix.random(1, 3).multiply(0.05));
            dense2.weights = dense2.weights.add(Matrix.random(3, 3).multiply(0.05));
            dense2.biases = dense2.biases.add(Matrix.random(1, 3).multiply(0.05));

            dense1.forward(X);
            activation1.forward(dense1.output);
            dense2.forward(activation1.output());
            activation2.forward(dense2.output);

            double loss = loss_function.calculate(loss_function.forward(activation2.output(), y));
            Matrix prediction = activation2.output().argmax(1).transpose();
            double accuracy = Matrix.bitwiseCompare(prediction, y).mean();
            if (loss < lowest_loss) {
                System.out.println(String.format("New set of weights found, iteration: %d loss: %s acc: %s", i, loss, accuracy));
                best_dense1_weights = new Matrix(dense1.weights);
                best_dense1_biases = new Matrix(dense1.biases);
                best_dense2_weights = new Matrix(dense2.weights);
                best_dense2_biases = new Matrix(dense2.biases);
                lowest_loss = loss;
            } else {
                dense1.weights = new Matrix(best_dense1_weights);
                dense1.biases = new Matrix(best_dense1_biases);
                dense2.weights = new Matrix(best_dense2_weights);
                dense2.biases = new Matrix(best_dense2_biases);
            }

        }
        //System.out.println(String.format("lowest loss %f", lowest_loss));
//        Matrix a = new Matrix(3,4,new double[][]{{1,2,3,4}, {5,6,7,8},{9,10,11,12}});
//        Matrix b = new Matrix(3,4,new double[][]{{11,22,33,44}, {55,66,77,88},{99,101,111,121}});
//        Matrix m = new Matrix(a.subtract(a.max(1)).exp());
//        System.out.println(m.sum(1));

    }

}
