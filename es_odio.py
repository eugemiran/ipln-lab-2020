from constants import MODEL_TYPES
import data
import model

def main():
    datasets = data.Data()
    model_type = MODEL_TYPES["SIMPLE"] # HERE GOES THE MODEL WE CONFIGURED IN CORSS VALIDATION
    m = model.Model(model_type=model_type, train_dataset=datasets.train, val_dataset=datasets.test)
    m.train()
    m.eval()

if __name__ == "__main__":
    main()