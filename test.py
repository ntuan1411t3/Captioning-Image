from PIL import Image
from utils import build_model, predict_captions, beam_search_predictions
import os


def main():
    pass


if __name__ == '__main__':
    main()
    weight_file = "model4.h5"
    final_model = build_model(weight_file)

    images = os.listdir("test_image/")
    # image_path = "test_image/47871819_db55ac4699.jpg"
    for im in images:
        if ".jpg" not in im:
            continue
        image_path = "test_image/" + im
        Image.open(image_path)
        print(image_path)
        print('Normal Max search:', predict_captions(final_model, image_path))
        print('Beam Search, k=3:', beam_search_predictions(
            final_model, image_path, beam_index=3))
        print('Beam Search, k=5:', beam_search_predictions(
            final_model, image_path, beam_index=5))
        print('Beam Search, k=7:', beam_search_predictions(
            final_model, image_path, beam_index=7))
