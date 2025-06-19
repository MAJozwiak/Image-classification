from model import build_model
from utils import get_data_generators


def main():
    train_dir = 'dataset/train'
    test_dir = 'dataset/test'
    img_size = (224, 224)
    batch_size = 32
    epochs = 10

    train_gen, test_gen = get_data_generators(train_dir, test_dir, img_size, batch_size)

    model = build_model(input_shape=(img_size[0], img_size[1], 3))

    model.fit(
        train_gen,
        epochs=epochs,
        validation_data=test_gen
    )

    loss, accuracy = model.evaluate(test_gen)
    print(f"{accuracy:.4f}")


if __name__ == '__main__':
    main()
