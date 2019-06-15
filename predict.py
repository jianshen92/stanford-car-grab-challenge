from fastai.vision import *
from pathlib import Path

defaults.device = torch.device('cpu')

FILE_PATH = #Insert path with image files
path = Path(__file__).parent
model_file = "best-performing-model.pkl"
learner = load_learner(path, model_file)


def predict_one_image(img_path, learner):
    img_path = Path(img_path)
    image = open_image(img_path)
    pred, label, prob = learn.predict(image)
    probability = prob[label].item()
    return pred, probability


def generate_csv_for_test_data(img_path, learner, output_fpath="./test.csv"):
    img_path = Path(img_path)
    fname_list = [path.parts[-1] for path in img_path.ls()]
    
    test_image_list = ImageList.from_folder(img_path)
    learner.data.add_test(test_image_list)
    
    pred, _ = learn.get_preds(ds_type=DatasetType.Test)
    probability, index = torch.max(pred, dim=1)
    
    classes_list = learn.data.classes
    prediction_list = [classes_list[label] for label in index]
    
    csv_df = pd.DataFrame(
    {'fname': fname_list,
     'prediction': prediction_list,
     'probability' : probability.tolist()
    })
    
    df.to_csv(path_or_buf=output_fpath)


generate_csv_for_test_data(FILE_PATH, learner, "./test.csv")
