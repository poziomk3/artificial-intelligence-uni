import os
import random
from collections import Counter

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import torch
from torch import nn

from lab5.main import FaceVerificationMLP, facenet, get_embedding, mtcnn, get_diff_vector, augment_image

def load_identity_map(identity_txt_path):
    """
    Ładuje słownik: {filename: identity_int}
    """
    identity_map = {}
    with open(identity_txt_path) as f:
        for line in f:
            fname, identity = line.strip().split()
            identity_map[fname] = int(identity)
    return identity_map


def make_pairs(identity_map, n_same=100, n_diff=100, exclude_files=None):
    if exclude_files is None:
        exclude_files = set()
    identity_to_files = {}
    for fname, ident in identity_map.items():
        identity_to_files.setdefault(ident, []).append(fname)

    # FILTR tylko osoby z ≥2 zdjęciami
    identity_with_multiple = {k: v for k, v in identity_to_files.items() if len(v) >= 2}

    # --- SAME PAIRS ---
    same_pairs = []
    all_possible_same = []
    for ident, files in identity_with_multiple.items():
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                all_possible_same.append((files[i], files[j], 1))
    random.shuffle(all_possible_same)
    same_pairs = all_possible_same[:min(len(all_possible_same), n_same)]

    # --- DIFF PAIRS ---
    all_idents = list(identity_to_files.keys())
    diff_pairs = []
    while len(diff_pairs) < n_diff:
        id1, id2 = random.sample(all_idents, 2)
        files1 = identity_to_files[id1]
        files2 = identity_to_files[id2]
        if files1 and files2:
            f1 = random.choice(files1)
            f2 = random.choice(files2)
            diff_pairs.append((f1, f2, 0))

    return same_pairs + diff_pairs



def prepare_dataset(pairs, get_diff_vector, images_dir):
    X = []
    y = []
    skipped = 0
    for f1, f2, label in pairs:
        img1_path = os.path.join(images_dir, f1)
        img2_path = os.path.join(images_dir, f2)
        try:
            diff = get_diff_vector(img1_path, img2_path)
            X.append(diff.squeeze(0))
            y.append(label)
        except Exception as e:
            # print(f"Error on pair ({f1}, {f2}): {e}")
            skipped += 1
    print(f"prepare_dataset: Wykorzystano {len(y)}/{len(pairs)} par (pominięto {skipped})")
    return torch.stack(X), torch.tensor(y)


### ---- 4.Funkcja do testowania i metryk ----
def evaluate(model, pairs, get_diff_vector, images_dir):
    model.eval()
    y_true = []
    y_pred = []
    for f1, f2, label in pairs:
        img1_path = os.path.join(images_dir, f1)
        img2_path = os.path.join(images_dir, f2)
        try:
            diff = get_diff_vector(img1_path, img2_path)
            output = model(diff)
            _, pred = torch.max(output, 1)
            y_true.append(label)
            y_pred.append(pred.item())
        except Exception as e:
            print(f"Error on pair ({f1}, {f2}): {e}")
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1s = f1_score(y_true, y_pred)
    return acc, prec, rec, f1s


def prepare_augmented_dataset(pairs, get_diff_vector, images_dir, augment_type="gaussian_noise"):
    """
    Jak prepare_dataset, ale pierwszy obrazek w parze jest zaburzony
    """
    from PIL import Image
    X = []
    y = []
    for f1, f2, label in pairs:
        img1_path = os.path.join(images_dir, f1)
        img2_path = os.path.join(images_dir, f2)
        try:
            # Załaduj, zaburz img1
            img1 = Image.open(img1_path).convert('RGB')
            img1_aug = augment_image(img1, augment_type=augment_type)
            # Zamień img1_aug na embedding
            face = mtcnn(img1_aug)
            if face is None:
                raise ValueError(f"No face detected in {img1_path}")
            emb1 = facenet(face.unsqueeze(0))
            emb2 = get_embedding(img2_path)
            diff = torch.abs(emb1 - emb2)
            X.append(diff.squeeze(0))
            y.append(label)
        except Exception as e:
            print(f"Error on pair ({f1}, {f2}): {e}")
    return torch.stack(X), torch.tensor(y)


def experiment_train_sizes(train_sizes, test_pairs_n=200, images_dir="img_align_celeba\\img_align_celeba",
                           identity_txt='identity_CelebA.txt', get_diff_vector=get_diff_vector):
    identity_map = load_identity_map(identity_txt)
    results = []

    test_pairs = make_pairs(identity_map, n_same=test_pairs_n // 2, n_diff=test_pairs_n // 2)
    test_pair_keys = set(frozenset((a, b)) for a, b, _ in test_pairs)
    test_files = set([f for f, _, _ in test_pairs] + [f2 for _, f2, _ in test_pairs])

    for n_train in train_sizes:
        raw_train_pairs = make_pairs(identity_map, n_same=n_train // 2, n_diff=n_train // 2, exclude_files=test_files)
        train_pairs = []
        train_pair_keys = set()
        for a, b, l in raw_train_pairs:
            key = frozenset((a, b))
            if key not in test_pair_keys and key not in train_pair_keys:
                train_pairs.append((a, b, l))
                train_pair_keys.add(key)
        X_train, y_train = prepare_dataset(train_pairs, get_diff_vector, images_dir)

        model = FaceVerificationMLP()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for epoch in range(20):
            model.train()
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()
        acc, prec, rec, f1s = evaluate(model, test_pairs, get_diff_vector, images_dir)
        print(f"Train size {n_train}: acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1s:.3f}")
        results.append((n_train, acc, prec, rec, f1s))
    return results


def experiment_learning_rates(lrs, train_n=1000, test_pairs_n=200, images_dir="img_align_celeba\\img_align_celeba",
                             identity_txt='identity_CelebA.txt', get_diff_vector=get_diff_vector):
    identity_map = load_identity_map(identity_txt)
    random.seed(123)
    test_pairs = make_pairs(identity_map, n_same=test_pairs_n // 2, n_diff=test_pairs_n // 2)
    test_pair_keys = set(frozenset((a, b)) for a, b, _ in test_pairs)
    test_files = set([f for f, _, _ in test_pairs] + [f2 for _, f2, _ in test_pairs])

    raw_train_pairs = make_pairs(identity_map, n_same=train_n // 2, n_diff=train_n // 2, exclude_files=test_files)
    train_pairs = []
    train_pair_keys = set()
    for a, b, l in raw_train_pairs:
        key = frozenset((a, b))
        if key not in test_pair_keys and key not in train_pair_keys:
            train_pairs.append((a, b, l))
            train_pair_keys.add(key)

    X_train, y_train = prepare_dataset(train_pairs, get_diff_vector, images_dir)

    results = []

    for lr in lrs:
        model = FaceVerificationMLP()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        epochs = 20  # liczba epok
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()
        acc, prec, rec, f1s = evaluate(model, test_pairs, get_diff_vector, images_dir)
        print(f"LR={lr:.5f}: acc={acc:.3f} prec={prec:.3f} rec={rec:.3f} f1={f1s:.3f}")
        results.append((lr, acc, prec, rec, f1s))
    return results



def experiment_num_epochs(epoch_list, train_n=1000, test_pairs_n=200, lr=0.001,
                         images_dir="img_align_celeba\\img_align_celeba",
                         identity_txt='identity_CelebA.txt', get_diff_vector=get_diff_vector):
    identity_map = load_identity_map(identity_txt)
    random.seed(123)
    test_pairs = make_pairs(identity_map, n_same=test_pairs_n // 2, n_diff=test_pairs_n // 2)
    test_pair_keys = set(frozenset((a, b)) for a, b, _ in test_pairs)
    test_files = set([f for f, _, _ in test_pairs] + [f2 for _, f2, _ in test_pairs])

    raw_train_pairs = make_pairs(identity_map, n_same=train_n // 2, n_diff=train_n // 2, exclude_files=test_files)
    train_pairs = []
    train_pair_keys = set()
    for a, b, l in raw_train_pairs:
        key = frozenset((a, b))
        if key not in test_pair_keys and key not in train_pair_keys:
            train_pairs.append((a, b, l))
            train_pair_keys.add(key)

    X_train, y_train = prepare_dataset(train_pairs, get_diff_vector, images_dir)

    results = []
    for epochs in epoch_list:
        model = FaceVerificationMLP()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()
        acc, prec, rec, f1s = evaluate(model, test_pairs, get_diff_vector, images_dir)
        results.append((epochs, acc, prec, rec, f1s))
    return results


def grid_search_best_model(train_sizes, lrs, epochs_list,
                           test_pairs_n=200,
                           images_dir="img_align_celeba\\img_align_celeba",
                           identity_txt='identity_CelebA.txt',
                           get_diff_vector=get_diff_vector):
    from collections import Counter
    import matplotlib.pyplot as plt

    identity_map = load_identity_map(identity_txt)
    random.seed(200002)
    # Stały test
    test_pairs = make_pairs(identity_map, n_same=test_pairs_n // 2, n_diff=test_pairs_n // 2)
    test_pair_keys = set(frozenset((a, b)) for a, b, _ in test_pairs)
    test_files = set([f for f, _, _ in test_pairs] + [f2 for _, f2, _ in test_pairs])

    best_result = None
    best_config = None
    all_results = []
    for train_n in train_sizes:
        # Stały train dla tego rozmiaru
        raw_train_pairs = make_pairs(identity_map, n_same=train_n // 2, n_diff=train_n // 2, exclude_files=test_files)
        train_pairs = []
        train_pair_keys = set()
        for a, b, l in raw_train_pairs:
            key = frozenset((a, b))
            if key not in test_pair_keys and key not in train_pair_keys:
                train_pairs.append((a, b, l))
                train_pair_keys.add(key)
        print(f"Train size {train_n}, class distribution: {Counter([l for _, _, l in train_pairs])}")

        X_train, y_train = prepare_dataset(train_pairs, get_diff_vector, images_dir)
        for lr in lrs:
            for epochs in epochs_list:
                print(f"== Train_n={train_n} lr={lr} epochs={epochs} ==")
                model = FaceVerificationMLP()
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                for epoch in range(epochs):
                    model.train()
                    optimizer.zero_grad()
                    output = model(X_train)
                    loss = criterion(output, y_train)
                    loss.backward()
                    optimizer.step()
                acc, prec, rec, f1s = evaluate(model, test_pairs, get_diff_vector, images_dir)
                print(f"ACC={acc:.3f} PREC={prec:.3f} REC={rec:.3f} F1={f1s:.3f}")
                result = (acc, prec, rec, f1s)
                config = (train_n, lr, epochs)
                all_results.append((config, result))
                # Najlepszy wg F1 (możesz wybrać acc)
                if best_result is None or f1s > best_result[3]:
                    best_result = result
                    best_config = config

    print("\n=== NAJLEPSZA KONFIGURACJA ===")
    print(f"train_size={best_config[0]}, lr={best_config[1]}, epochs={best_config[2]}")
    print(f"ACC={best_result[0]:.3f} PREC={best_result[1]:.3f} REC={best_result[2]:.3f} F1={best_result[3]:.3f}")
    return all_results, best_config, best_result


if __name__ == "__main__":
    # experiment_train_sizes([10, 100,500,1000,5000], test_pairs_n=200)
    # experiment_learning_rates([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1], train_n=1000, test_pairs_n=200)
    # experiment_num_epochs([5, 10, 20, 30, 50, 100], train_n=1000, test_pairs_n=200, lr=0.001)
    all_results, best_config, best_result = grid_search_best_model(
            train_sizes=[500, 1000, 2000],  # Możesz zmienić na większe jeśli masz RAM!
            lrs=[0.001, 0.005, 0.01],
            epochs_list=[20, 50, 100]
        )
