import com.github.sh0nk.matplotlib4j.PythonExecutionException;

import java.io.IOException;

public class LayerDense {
    private Matrix weights;
    private Matrix biases;
    private Matrix output;
    private Matrix inputs;
    private Matrix d_weights;
    private Matrix d_biases;
    private Matrix d_inputs;

    protected LayerDense(int n_inputs, int n_neurons) {
        this.weights = Matrix.random(n_inputs, n_neurons).multiply(0.01);
        this.biases = new Matrix(1, n_neurons);
    }

    protected void backwards(Matrix d_values) {
        this.d_weights = this.d_inputs.transpose().dot(d_values); //Gradients on parameters
        this.d_biases = d_values.sum(0); //Gradients on parameters
        this.d_inputs = d_values.dot(this.weights.transpose()); //Gradient on values

    }

    /**
     * Calculate output values from inputs, weights and biases
     *
     * @param inputs (Matrix object).
     */
    protected void forward(Matrix inputs) {
        this.inputs = new Matrix(inputs);
        this.output = addBias(inputs.dot(this.weights), this.biases);
    }


    protected static Matrix addBias(Matrix B, Matrix V) throws IndexOutOfBoundsException {
        if (B.getColumns() != V.getColumns())
            throw new IndexOutOfBoundsException(String.format("Matrices has different dimensions (%s,%s) By (%s,%s)", B.getRows(), B.getColumns(), V.getRows(), V.getColumns()));
        else
            B.add(V);

        return B;
//        for (int i = 0; i < B.getRows(); i++) {
//            for (int j = 0; j < B.getColumns(); j++)
//                B.setValue(i, j, V.getValue(0, j) + B.getValue(i, j));
//        }

    }

    public static void main(String[] args) throws PythonExecutionException, IOException {
//        Matrix X = new Dataset().getVertical_data();
//        Matrix y = new Dataset().getVertical_classes();
//        LayerDense dense1 = new LayerDense(2, 3);
//        LayerDense dense2 = new LayerDense(3, 3);
//
//        Activation activation1 = new Activation_ReLU();
//        Activation activation2 = new Activation_SoftMax();
//
//        Loss_CategoricalCrossentropy loss_function = new Loss_CategoricalCrossentropy();
//
//
//        double lowest_loss = 9999999;
//        Matrix best_dense1_weights = new Matrix(dense1.weights);
//        Matrix best_dense1_biases = new Matrix(dense1.biases);
//        Matrix best_dense2_weights = new Matrix(dense2.weights);
//        Matrix best_dense2_biases = new Matrix(dense2.biases);
//
//        for (int i = 0; i < 1; i++) {
//            dense1.weights = dense1.weights.add(Matrix.random(2, 3).multiply(0.05));
//            dense1.biases = dense1.biases.add(Matrix.random(1, 3).multiply(0.05));
//            dense2.weights = dense2.weights.add(Matrix.random(3, 3).multiply(0.05));
//            dense2.biases = dense2.biases.add(Matrix.random(1, 3).multiply(0.05));
//
//            dense1.forward(X);
//            activation1.forward(dense1.output);
//            dense2.forward(activation1.output());
//            activation2.forward(dense2.output);
//
//            double loss = loss_function.calculate(loss_function.forward(activation2.output(), y));
//            Matrix prediction = activation2.output().argmax(1).transpose();
//            double accuracy = Matrix.bitwiseCompare(prediction, y).mean();
//            if (loss < lowest_loss) {
//                System.out.println(String.format("New set of weights found, iteration: %d loss: %s acc: %s", i, loss, accuracy));
//                best_dense1_weights = new Matrix(dense1.weights);
//                best_dense1_biases = new Matrix(dense1.biases);
//                best_dense2_weights = new Matrix(dense2.weights);
//                best_dense2_biases = new Matrix(dense2.biases);
//                lowest_loss = loss;
//            } else {
//                dense1.weights = new Matrix(best_dense1_weights);
//                dense1.biases = new Matrix(best_dense1_biases);
//                dense2.weights = new Matrix(best_dense2_weights);
//                dense2.biases = new Matrix(best_dense2_biases);
//            }
//
//        }
//        //System.out.println(String.format("lowest loss %f", lowest_loss));
////        Matrix a = new Matrix(3,4,new double[][]{{1,2,3,4}, {5,6,7,8},{9,10,11,12}});
////        Matrix b = new Matrix(3,4,new double[][]{{11,22,33,44}, {55,66,77,88},{99,101,111,121}});
////        Matrix m = new Matrix(a.subtract(a.max(1)).exp());
////        System.out.println(m.sum(1));
//
//        Matrix x = new Matrix(1, 3, new double[][]{{1.0, -2.0, 3.0}});
//        Matrix w = new Matrix(1, 3, new double[][]{{-3.0, -1.0, 2.0}});
//        double b = 1.0;
//        double z = x.getValue(0, 0) * w.getValue(0, 0) +
//                x.getValue(0, 1) * w.getValue(0, 1) +
//                x.getValue(0, 2) * w.getValue(0, 2) + b;
//        double _y = Math.max(z, 0);
//        double dvalue = 1.0;
//        double drelu_dz = dvalue * (z > 0 ? 1.0 : 0);
//
//        double dsum_dxw0 = 1;
//        double dsum_dxw1 = 1;
//        double dsum_dxw2 = 1;
//        double dsum_db = 1;
//
//        double drelu_dxw0 = drelu_dz * dsum_dxw0;
//        double drelu_dxw1 = drelu_dz * dsum_dxw1;
//        double drelu_dxw2 = drelu_dz * dsum_dxw2;
//        double drelu_db = drelu_dz * dsum_db;
//
//        double dmul_dx0 = w.getValue(0, 0);
//        double dmul_dx1 = w.getValue(0, 1);
//        double dmul_dx2 = w.getValue(0, 2);
//
//        double dmul_dw0 = x.getValue(0, 0);
//        double dmul_dw1 = x.getValue(0, 1);
//        double dmul_dw2 = x.getValue(0, 2);
//
//        double drelu_dx0 = drelu_dxw0 * dmul_dx0;
//        double drelu_dw0 = drelu_dxw0 * dmul_dw0;
//        double drelu_dx1 = drelu_dxw1 * dmul_dx1;
//        double drelu_dw1 = drelu_dxw1 * dmul_dw1;
//        double drelu_dx2 = drelu_dxw2 * dmul_dx2;
//        double drelu_dw2 = drelu_dxw2 * dmul_dw2;
//
//        Matrix dvalues = new Matrix(3, 3, new double[][]{
//                {1.0, 1.0, 1.0},
//                {2.0, 2.0, 2.0},
//                {3.0, 3.0, 3.0}});
//
//        Matrix weights = new Matrix(3, 4, new double[][]{
//                {0.2, 0.8, -0.5, 1.0},
//                {0.5, -0.91, 0.26, -0.5},
//                {-0.26, -0.27, 0.17, 0.87}}).transpose();
//
//        Matrix inputs = new Matrix(3, 4, new double[][]{
//                {1.0, 2.0, 3.0, 2.5},
//                {2.0, 5.0, -1.0, 2.0},
//                {-1.5, 2.7, 3.3, -0.8}});
//
//        Matrix biases = new Matrix(1, 3, new double[][]{
//                {2.0, 3.0, 0.5}});
//
//        Matrix dbiases = dvalues.sum(0);
//
//        Matrix layer_output = addBias(inputs.dot(weights), biases);
//        Matrix relu_outputs = Matrix.maximum(layer_output, 0);
//
//        Matrix drelu = new Matrix(relu_outputs);
//
//        for (int i = 0; i < drelu.getRows(); i++) {
//            for (int j = 0; j < drelu.getColumns(); j++) {
//                if (layer_output.getValue(i, j) <= 0)
//                    drelu.setValue(i, j, 0.0);
//                else
//                    drelu.setValue(i, j, layer_output.getValue(i, j));
//            }
//        }
//        Matrix dinputs = drelu.dot(weights.transpose());
//        Matrix dweights = inputs.transpose().dot(drelu);
//
//        weights = weights.add(dweights.multiply(-0.001));
//        biases = biases.add(dbiases.multiply(-0.001));
//
//        System.out.println(weights);
//        System.out.println(biases);
    }

}
