public class Activation_Softmax_Loss_CategoricalCrossEntropy {
    private Activation activation;
    private Loss loss;
    private Matrix output;

    protected Activation_Softmax_Loss_CategoricalCrossEntropy() {
        this.activation = new Activation_SoftMax();
        this.loss = new Loss_CategoricalCrossEntropy();
    }


    public double forward(Matrix inputs, Matrix y_true) {
        this.activation.forward(inputs);
        this.output = this.activation.output();

        Matrix forward_matrix = this.loss.forward(this.output, y_true);
        return this.loss.calculate(forward_matrix);
    }

    public void backward(Matrix d_values) {
        int samples = d_values.getRows();
    }

    @Override
    public Matrix output() {
        return null;
    }
}
