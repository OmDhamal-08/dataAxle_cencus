# Databricks notebook source
# Very small XGBoost script for free-tier Databricks.
# Run only if your cluster has xgboost installed. Keep row_limit small.

from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, Imputer, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import mlflow
from datetime import datetime

# COMMAND ----------

try:
    from xgboost.spark import SparkXGBClassifier
except Exception as exc:
    raise ImportError(
        "XGBoost is not available on this cluster. For free tier, use Logistic Regression or Decision Tree first."
    ) from exc

# COMMAND ----------

dbutils.widgets.text("source_table", "workshop.default.donation_data_v1")
dbutils.widgets.text("experiment_name", "/Shared/census_free_tier_xgboost")
dbutils.widgets.text("row_limit", "3000")
dbutils.widgets.dropdown("log_model_artifact", "false", ["false", "true"])

source_table = dbutils.widgets.get("source_table")
experiment_name = dbutils.widgets.get("experiment_name")
row_limit = int(dbutils.widgets.get("row_limit"))
log_model_artifact = dbutils.widgets.get("log_model_artifact").lower() == "true"

mlflow.set_experiment(experiment_name)

print("Source table:", source_table)
print("Row limit:", row_limit)
print("Log model artifact:", log_model_artifact)

# COMMAND ----------

df = spark.table(source_table)

label_col = "income"

numeric_features = [
    "age",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

# Fewer categorical columns keeps the feature vector and XGBoost memory lower.
categorical_features = [
    "workclass",
    "education_level",
    "marital_status",
    "occupation",
    "relationship",
    "sex",
]

model_df = df.select([label_col] + numeric_features + categorical_features)
model_df = model_df.filter(F.col(label_col).isin("<=80k", ">80k"))

for column_name in categorical_features:
    model_df = model_df.withColumn(column_name, F.coalesce(F.col(column_name), F.lit("Unknown")))

if row_limit > 0:
    model_df = model_df.limit(row_limit)

display(model_df.groupBy(label_col).count())

# COMMAND ----------

train_df, test_df = model_df.randomSplit([0.8, 0.2], seed=42)

print("Train rows:", train_df.count())
print("Test rows:", test_df.count())

# COMMAND ----------

label_indexer = StringIndexer(
    inputCol=label_col,
    outputCol="label",
    handleInvalid="skip",
    stringOrderType="alphabetAsc",
)

numeric_imputer = Imputer(
    inputCols=numeric_features,
    outputCols=[f"{c}_imputed" for c in numeric_features],
    strategy="median",
)

categorical_indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
    for c in categorical_features
]

feature_assembler = VectorAssembler(
    inputCols=[f"{c}_imputed" for c in numeric_features] + [f"{c}_idx" for c in categorical_features],
    outputCol="features",
)

xgb = SparkXGBClassifier(
    features_col="features",
    label_col="label",
    prediction_col="prediction",
    probability_col="probability",
    raw_prediction_col="rawPrediction",
    max_depth=3,
    n_estimators=20,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    num_workers=1,
    seed=42,
)

pipeline = Pipeline(
    stages=[
        label_indexer,
        numeric_imputer,
        *categorical_indexers,
        feature_assembler,
        xgb,
    ]
)

# COMMAND ----------

auc_evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
accuracy_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
precision_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
recall_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
f1_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")


def evaluate(predictions):
    return {
        "accuracy": accuracy_evaluator.evaluate(predictions),
        "precision": precision_evaluator.evaluate(predictions),
        "recall": recall_evaluator.evaluate(predictions),
        "f1_score": f1_evaluator.evaluate(predictions),
        "roc_auc": auc_evaluator.evaluate(predictions),
    }


# COMMAND ----------

with mlflow.start_run(run_name="free_tier_xgboost_tiny"):
    mlflow.log_param("source_table", source_table)
    mlflow.log_param("row_limit", row_limit)
    mlflow.log_param("model_type", "SparkXGBClassifier")
    mlflow.log_param("encoding", "StringIndexer compact encoding")
    mlflow.log_param("max_depth", 3)
    mlflow.log_param("n_estimators", 20)
    mlflow.log_param("num_workers", 1)
    mlflow.log_param("run_timestamp", datetime.utcnow().isoformat())

    model = pipeline.fit(train_df)
    train_predictions = model.transform(train_df)
    test_predictions = model.transform(test_df)

    train_metrics = evaluate(train_predictions)
    test_metrics = evaluate(test_predictions)

    for metric_name, metric_value in train_metrics.items():
        mlflow.log_metric(f"train_{metric_name}", metric_value)

    for metric_name, metric_value in test_metrics.items():
        mlflow.log_metric(f"test_{metric_name}", metric_value)

    mlflow.log_metric("roc_auc_overfit_gap", train_metrics["roc_auc"] - test_metrics["roc_auc"])

    if log_model_artifact:
        import mlflow.spark

        mlflow.spark.log_model(model, artifact_path="model")

print("Train metrics:", train_metrics)
print("Test metrics:", test_metrics)

# COMMAND ----------

display(test_predictions.groupBy("label", "prediction").count().orderBy("label", "prediction"))
