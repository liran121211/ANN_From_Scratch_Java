public interface Loss {
    public Matrix forward(Matrix output, Matrix y_true);

    public void backward(Matrix output, Matrix y_true);

    public Matrix d_inputs();

    public double calculate(Matrix output);
}
