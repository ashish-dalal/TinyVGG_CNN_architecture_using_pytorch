import os
import torch
import data_setup, engine, model_builder, utils

from torchvision import transforms

NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

def main():
    # setup directories
    train_dir = "data/pizza_steak_sushi/train"
    test_dir = "data/pizza_steak_sushi/test"

    # setup target device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # create transforms
    train_transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor()
    ])

    # create dataloader with data_setup.py
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloader(
        train_dir = train_dir,
        test_dir = test_dir,
        train_transform = train_transform,
        test_transform = test_transform,
        batch_size = BATCH_SIZE
    )

    # create model with help from model_builder.py
    model = model_builder.TinyVGG(
        input_shape=3,
        hidden_units=HIDDEN_UNITS,
        output_shape=len(class_names)
    ).to(device)

    # set up loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=NUM_EPOCHS,
        device=device 
    )

    # Save the model with utils.py
    utils.save_model(
        model=model,
        target_dir="models",
        model_name="05_TinyVGG_model_1.pth"
    )

if __name__ == '__main__':
    main()