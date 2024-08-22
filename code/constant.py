dataset_list = [
    "100leaves",
    "handwritten_2v",
    "msrcv1_6v",
    "orl",
    "animal",
    "ADNI_3classes",
    "3sources_incomplete",
]
fixed_args = {
    "h_dim": 50,
    "epochs": 1000,
    "lr": 1e-3,
    "k_find": 20,
    "print_loss_freq": 50,
    "construct_w_way": "sklearn",  # sklearn or faiss
}
mr05_dataset_args_map = {
    "100leaves": {
        "k": 5,
        "lam": 100,
    },
    "handwritten_2v": {
        "k": 5,
        "lam": 0.01,
    },
    "msrcv1_6v": {
        "k": 5,
        "lam": 10,
    },
    "orl": {
        "k": 5,
        "lam": 10,
    },
    "ADNI_3classes": {
        "k": 6,
        "lam": 1,
    },
    "3sources_incomplete": {
        "k": 7,
        "lam": 10,
    },
    "animal": {
        "k": 5,
        "lam": 1,
    },
}
