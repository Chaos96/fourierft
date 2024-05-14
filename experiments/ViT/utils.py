def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        if 'classifier' in name:
            continue
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
    return trainable_params, all_param

def subset_of(dataset, n_classes):
    """
    Returns a subset of the dataset with n_classes.
    """
    unique_labels = dataset.unique("label")
    labels = unique_labels[:n_classes]
    return dataset.filter(lambda example: example["label"] in labels)