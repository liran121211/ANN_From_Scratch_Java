public class Activation_SoftMax implements Activation {
    private Matrix outputs;

    @Override
    public Matrix output() {
        return outputs;
    }

    @Override
    public void forward(Matrix inputs) {
        this.outputs = new Matrix(inputs.getRows(), inputs.getColumns());
        double sum = 0;
        for (int i = 0; i < inputs.getRows(); i++) {
            for (int j = 0; j < inputs.getColumns(); j++) {
                this.outputs.setValue(i, j, Math.pow(Math.E, inputs.getValue(i, j)));
                sum += this.outputs.getValue(i, j);
            }
        }
        for (int i = 0; i < inputs.getRows(); i++)
            for (int j = 0; j < inputs.getColumns(); j++)
                this.outputs.setValue(i, j, this.outputs.getValue(i, j) / sum);

    }


}
