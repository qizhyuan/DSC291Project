

def accuracy(predictions, labels):
    assert len(predictions) == len(labels)

    correct = 0
    for pred, label in zip(predictions, labels):
        if pred > 0.5 and label == 1:
            correct += 1
        elif pred < 0.5 and label == 0:
            correct += 1

    return correct / len(predictions)

