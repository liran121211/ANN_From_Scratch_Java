public class Activation_SoftMax implements Activation {
    private Matrix outputs;
    private Matrix d_inputs;

    @Override
    public void forward(Matrix inputs) {
        this.outputs = new Matrix(inputs.getRows(), inputs.getColumns());
        Matrix np_max = new Matrix(inputs.max(1));
        Matrix subtract = new Matrix(inputs.subtract(np_max));
        Matrix exp_values = new Matrix(subtract.exp());

        Matrix np_sum = new Matrix(exp_values.sum(1));
        this.outputs = new Matrix(exp_values.divide(np_sum));
    }

    @Override
    public void backward(Matrix d_values) {
        this.d_inputs = new Matrix(d_values.getRows(), d_values.getColumns()); // Create zeros array

        for (int i = 0; i < d_values.getRows(); i++) { // like enumerate(zip(this.output,d_values))
            Matrix single_output = this.outputs.getRow(i).reshape(-1, 1); // Flatten output array
            Matrix jacobian_matrix = Matrix.diagflat(single_output).subtract(single_output.dot(single_output.transpose())); // Calculate Jacobian matrix of the output

            // Calculate sample-wise gradient
            // and add it to the array of sample gradients
            Matrix vector = jacobian_matrix.dot(d_values.getRow(i).transpose());
            for (int j = 0; j < d_inputs.getRows(); j++) {
                d_inputs.setValue(j, i, vector.getValue(j, 0));
            }
        }

    }

    public Matrix d_inputs() {
        return d_inputs;
    }

    @Override
    public Matrix output() {
        return outputs;
    }
}
