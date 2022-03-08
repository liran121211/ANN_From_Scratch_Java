class Loss_CategoricalCrossEntropy implements Loss {
    private Matrix d_inputs;

    @Override
    public void backward(Matrix d_values, Matrix y_true) {
        int samples = d_values.getRows();
        int labels = d_values.getColumns(); //Might not be useful??

        // If labels are sparse, turn them into one-hot vector
        if (y_true.getRows() == 1)
            y_true = Matrix.oneHotVector(y_true);

        // Calculate gradient
        this.d_inputs = y_true.divide(d_values).multiply(-1);

        // Normalize gradient
        this.d_inputs = this.d_inputs.divide(samples);

    }

    @Override
    public Matrix forward(Matrix y_pred, Matrix y_true) throws IndexOutOfBoundsException {
        int samples = y_pred.getRows();
        Matrix y_pred_clipped = Matrix.clip(y_pred, 1e-7, 1 - 1e-7);
        Matrix correct_confidences = new Matrix(samples, 1);

        if (y_true.getRows() == 1) { //1D array label data
            if (y_pred.getRows() == y_true.getColumns()) { // Check if Prediction size == Labels size
                for (int i = 0; i < samples; i++)
                    correct_confidences.setValue(i, 0, y_pred_clipped.getValue(i, (int) y_true.getValue(0, i)));
            } else //2D array label data
                throw new IndexOutOfBoundsException(String.format("Prediction Matrix (%s, %s) and Labels Matrix (, %s) has different shapes", y_pred.getRows(), y_pred.getColumns(), y_true.getColumns()));

        } else {
            if (y_pred.getRows() == y_true.getRows() && y_pred.getColumns() == y_true.getColumns())
                correct_confidences = y_pred_clipped.multiply(y_true);
            else
                throw new IndexOutOfBoundsException(String.format("Prediction Matrix (%s, %s) and Labels Matrix (%s, %s) has different shapes", y_pred.getRows(), y_pred.getColumns(), y_true.getRows(), y_true.getColumns()));
        }

        return correct_confidences.log().multiply(-1);
    }

    @Override
    public double calculate(Matrix output) {
        return output.mean();
    }

    @Override
    public Matrix d_inputs() {
        return d_inputs;
    }
}



