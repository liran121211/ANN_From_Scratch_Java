public interface Loss {
    public double calculate(Matrix output);
    public Matrix forward(Matrix output, Matrix y_true);
}
