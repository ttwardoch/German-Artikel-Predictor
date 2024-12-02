import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class GermanNounDataset(Dataset):
    def __init__(self, data, char_to_idx):
        self.data = data
        self.char_to_idx = char_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        index, word = self.data[idx]
        # Convert word to character indices
        word_indices = [self.char_to_idx[char] for char in word]

        return torch.tensor(word_indices, dtype=torch.long), index


def create_dataloaders(file_path="words_big.txt", data_fraction=1, test_size=0.2, batch_size=64):
    """
    Reads a file and forms dataloaders

    :param file_path:
    :param data_fraction:
    :param test_size:
    :param batch_size:
    :return:
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            word, numbers = line.split(maxsplit=1)
            num1, num2, num3 = map(int, numbers.split(','))

            if num1 != 0:
                data.append([0, word])
            if num2 != 0:
                data.append([1, word])
            if num3 != 0:
                data.append([2, word])

    # Shorten the dataset for quicker validation
    data = data[:int(len(data)*data_fraction)]

    # Print balance of classes
    counts = [0, 0, 0]
    for i in data:
        counts[i[0]] += 1
    print(f"Words with der: {counts[0]}, words with die: {counts[1]}, words with das: {counts[2]}")

    # Change letters into numbers
    chars = set("".join([word for _, word in data]))
    char_to_idx = {char: idx + 1 for idx, char in enumerate(sorted(chars))}
    char_to_idx['<pad>'] = 0  # Padding token

    # Dataset splitting and dataloader creation
    data_train, data_test = train_test_split(data, test_size=test_size)
    train_dataset = GermanNounDataset(data_train, char_to_idx)
    test_dataset = GermanNounDataset(data_test, char_to_idx)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=lambda x: collate_fn(x))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=lambda x: collate_fn(x))

    return train_dataloader, test_dataloader


# Collate function to pad sequences
def collate_fn(batch):
    words, indices = zip(*batch)
    max_len = max(len(w) for w in words)
    padded_words = torch.zeros(len(words), max_len, dtype=torch.long)

    for i, word in enumerate(words):
        padded_words[i, :len(word)] = word

    return padded_words, torch.tensor(indices)