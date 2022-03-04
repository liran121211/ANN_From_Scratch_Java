public class Activation_ReLU implements Activation{
    private Matrix outputs;

    @Override
    public Matrix output() {
        return outputs;
    }

    @Override
    public void forward(Matrix inputs) {
        this.outputs = new Matrix(inputs.getRows(), inputs.getColumns());
        for (int i = 0; i < inputs.getRows(); i++) {
            for (int j = 0; j < inputs.getColumns(); j++)
                this.outputs.setValue(i, j, Math.max(0, inputs.getValue(i, j)));
        }
    }


}
