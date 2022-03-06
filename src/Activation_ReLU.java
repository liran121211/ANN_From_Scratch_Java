public class Activation_ReLU implements Activation {
    private Matrix outputs;
    private Matrix inputs;
    private Matrix d_inputs;

    @Override
    public Matrix output() {
        return outputs;
    }

    public Matrix input() {
        return inputs;
    }

    public Matrix d_input() {
        return d_inputs;
    }

    @Override
    public void forward(Matrix inputs) {
        this.inputs = new Matrix(inputs);
        this.outputs = new Matrix(inputs.getRows(), inputs.getColumns());
        for (int i = 0; i < inputs.getRows(); i++) {
            for (int j = 0; j < inputs.getColumns(); j++)
                this.outputs.setValue(i, j, Math.max(0.0, inputs.getValue(i, j)));
        }
    }

    @Override
    public void backwards(Matrix d_values) {
        this.d_inputs = new Matrix(d_values);
        for (int i = 0; i < d_inputs.getRows(); i++) {
            for (int j = 0; j < d_inputs.getColumns(); j++) {
                if (this.inputs.getValue(i, j) <= 0)
                    this.d_inputs.setValue(i, j, 0);
                else
                    this.d_inputs.setValue(i, j, this.inputs.getValue(i, j));
            }
        }
    }
}
