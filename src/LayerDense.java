import javax.naming.NameNotFoundException;

public class LayerDense {
    private Matrix weights;
    private Matrix biases;
    private Matrix outputs;

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
        this.outputs = addBias(inputs.dot(this.weights), this.biases);
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

    public static void main(String[] args) {
        Matrix inputs = new Matrix(3, 4, new double[][]{{1.0, 2.0, 3.0, 2.5}, {2.0, 5.0, -1.0, 2.0}, {-1.5, 2.7, 3.3, -0.8}});
        Matrix y = new Matrix(4, 3, new double[][]{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 0}});
        LayerDense dense1 = new LayerDense(3, 3);
        LayerDense dense2 = new LayerDense(3, 3);

        Activation activation1 = new Activation_ReLU();
        Activation activation2 = new Activation_SoftMax();

        Loss_CategoricalCrossentropy loss_function = new Loss_CategoricalCrossentropy();

        dense1.forward(inputs.transpose());
        activation1.forward(dense1.outputs);
        dense2.forward(activation1.output());
        activation2.forward(dense2.outputs);
        System.out.println(activation2.output());
        System.out.println(loss_function.forward(activation2.output(),y));


//        Matrix softmax_outputs = new Matrix(3, 3, new double[][]{{0.7, 0.1, 5.2}, {0.1, 0.5, 0.4}, {0.02, 0.9, 0.08}});
//        Matrix class_targets = new Matrix(3, 3, new double[][]{{1, 0, 0}, {0, 1, 0}, {0, 1, 0}});
    }

}
