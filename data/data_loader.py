from torch.utils.data import DataLoader


def get_data_loader(
    train_dataset, valid_dataset, test_dataset, batch_size: int, num_worker: int
):
    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_worker, shuffle=True,
    )
    valid_data_loader = DataLoader(
        valid_dataset, batch_size=batch_size, num_workers=num_worker, shuffle=False,
    )
    test_data_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=num_worker, shuffle=False,
    )

    return train_data_loader, valid_data_loader, test_data_loader
