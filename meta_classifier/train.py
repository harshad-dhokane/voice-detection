import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from meta_classifier.model import FEATURE_NAMES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to numpy feature file")
    parser.add_argument("--labels", required=True, help="Path to numpy labels file")
    parser.add_argument("--output", default="meta_classifier/meta_model.joblib")
    args = parser.parse_args()

    features = np.load(args.data)
    labels = np.load(args.labels)

    x_train, x_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    score = model.score(x_val, y_val)
    print(f"Validation accuracy: {score:.4f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output)
    print(f"Saved model to {output}")


if __name__ == "__main__":
    main()
