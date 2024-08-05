
def data_generator(X, y, batch_size):
    num_samples = X.shape[0]
    while True:
        for offset in range(0, num_samples, batch_size):
            end = offset + batch_size
            batch_X = X[offset:end]
            batch_y = y[offset:end]

            yield batch_X, batch_y
