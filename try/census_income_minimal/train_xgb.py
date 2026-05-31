from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

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
    parser = training_parser("Train XGBoost income model.")
    args = parser.parse_args()

    X, y = load_training_data(args.input)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    pipeline = build_model_pipeline(
        XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.85,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        scale_numeric=False,
    )
    pipeline.fit(X_train, y_train_encoded)

    metrics = evaluate_classifier(pipeline, X_test, y_test, label_encoder=label_encoder)
    model_path = args.models_dir / "xgb_model.pkl"
    save_model_bundle(model_path, "xgboost", pipeline, metrics, label_encoder=label_encoder)
    print_training_result("xgboost", model_path, metrics)


if __name__ == "__main__":
    main()
