from constants import MODEL_TYPES
import data
import model

def main():
    datasets = data.Data()
    model_type = MODEL_TYPES["LSTM1"]  # HERE GOES THE MODEL WE CONFIGURED IN CORSS VALIDATION
    epochs = 50                        # HERE GOES THE EPOCHS WE CONFIGURED IN CORSS VALIDATION
    neurons = 128                      # HERE GOES THE NEURONS WE CONFIGURED IN CORSS VALIDATION
    dropout = 0.5                      # HERE GOES THE DROPOUT WE CONFIGURED IN CORSS VALIDATION
    batches_size = 128                 # HERE GOES THE BATCH_SIZE WE CONFIGURED IN CORSS VALIDATION
    m = model.Model(model_type=model_type, train_dataset=datasets.train, neurons=neurons, dropout=dropout, val_dataset=datasets.test)
    m.train(epochs, batches_size)
    m.eval()

if __name__ == "__main__":
    main()