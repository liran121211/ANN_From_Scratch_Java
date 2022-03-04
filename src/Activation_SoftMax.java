public class Activation_SoftMax implements Activation {
    private Matrix outputs;

    @Override
    public Matrix output() {
        return outputs;
    }

    @Override
    public void forward(Matrix inputs) {
        this.outputs = new Matrix(inputs.getRows(), inputs.getColumns());
        Matrix np_max = new Matrix(inputs.max(1));
        Matrix subtract = new Matrix(inputs.subtract(np_max));
        Matrix exp_values = new Matrix(subtract.exp());

        Matrix np_sum = new Matrix(exp_values.sum(1));
        Matrix divide = new Matrix(exp_values.divide(np_sum));
        this.outputs = divide;
    }


}
