from fastai.vision import *
from pathlib import Path

import fire

defaults.device = torch.device('cpu')

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


def generate_csv_for_test_data(img_path, learner=learner, output_fpath=OUTPUT_PATH):
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
    
    csv_df.to_csv(path_or_buf=output_fpath)


if __name__ == '__main__':
    fire.Fire()
