public interface Activation {
    public void forward(Matrix inputs);

    public void backwards(Matrix d_values);

    public Matrix output();
}
