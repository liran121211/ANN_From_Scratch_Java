public interface Activation {
    public void forward(Matrix inputs);
    public void backward(Matrix d_values);

    public Matrix output();
    public Matrix d_inputs();
}
