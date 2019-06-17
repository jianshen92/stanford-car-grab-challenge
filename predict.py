from fastai.vision import *
from pathlib import Path

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import fire

path = Path(__file__).parent
OUTPUT_PATH = "./test.csv" # Modify this with your output csv file

model_file = "best-model.pkl"
learner = load_learner(path, model_file)


def predict_one_image(img_path, learner=learner):
    img_path = Path(img_path)
    image = open_image(img_path)
    pred, label, prob = learner.predict(image)
    probability = prob[label].item()
    return pred, probability


def _predict_batch(img_path, learner):
    img_path = Path(img_path)
    fname_list = [path.parts[-1] for path in img_path.ls()]
    
    test_image_list = ImageList.from_folder(img_path)
    learner.data.add_test(test_image_list)
    
    pred, _ = learner.get_preds(ds_type=DatasetType.Test)
    probability, index = torch.max(pred, dim=1)
    
    classes_list = learner.data.classes
    prediction_list = [classes_list[label] for label in index]
     
    csv_df = pd.DataFrame(
    {'fname': fname_list,
     'prediction': prediction_list,
     'probability' : probability.tolist()
    })

    return csv_df 


def generate_csv_for_test_data(img_path, learner=learner, output_fpath=OUTPUT_PATH):
    print('Running Inference ...')
    test_df = _predict_batch(img_path, learner)
    test_df.to_csv(path_or_buf=output_fpath)
    print(f"Results saved to {output_fpath}")


def populate_csv_for_labelled_data(csv_path, img_path, learner=learner, output_fpath="./labelled.csv"):
    print('Running Inference ...')
    test_df = _predict_batch(img_path, learner)

    label_df = pd.read_csv(csv_path, usecols=["fname", "label"])

    new_df = label_df.join(test_df.set_index('fname'), on='fname')
    new_df.to_csv(path_or_buf=output_fpath)
    print(f"Results saved to {output_fpath}")

    print('Calculating Performance')
    print('-----------------------')
    accuracy = accuracy_score(new_df["label"].tolist(), new_df["prediction"].tolist())
    precision = precision_score(new_df["label"].tolist(), new_df["prediction"].tolist(), average="weighted")
    recall = recall_score(new_df["label"].tolist(), new_df["prediction"].tolist(), average="weighted")
    f1 = f1_score(new_df["label"].tolist(), new_df["prediction"].tolist(), average="weighted")

    print(f"Accuracy : {accuracy*100}% \nPrecision : {precision*100}% \nRecall : {recall*100}% \nf1 : {f1*100}%")


if __name__ == '__main__':
    fire.Fire()
