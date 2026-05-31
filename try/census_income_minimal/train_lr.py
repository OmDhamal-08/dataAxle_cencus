from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from common import (
    RANDOM_STATE,
    build_model_pipeline,
    evaluate_classifier,
    load_training_data,
    print_training_result,
    save_model_bundle,
    training_parser,
)


def main():
    parser = training_parser("Train Logistic Regression income model.")
    args = parser.parse_args()

    X, y = load_training_data(args.input)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = build_model_pipeline(
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        scale_numeric=True,
    )
    pipeline.fit(X_train, y_train)

    metrics = evaluate_classifier(pipeline, X_test, y_test)
    model_path = args.models_dir / "lr_model.pkl"
    save_model_bundle(model_path, "logistic_regression", pipeline, metrics)
    print_training_result("logistic_regression", model_path, metrics)


if __name__ == "__main__":
    main()
