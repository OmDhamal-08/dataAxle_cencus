# Databricks notebook source
# Decision Tree model for census income prediction.
# Run this after the curated Delta table is created.

from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    Imputer,
    VectorAssembler,
    StandardScaler,
)
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import mlflow
import mlflow.spark
from datetime import datetime

# COMMAND ----------

dbutils.widgets.text("source_table", "workshop.default.donation_data_v1")
dbutils.widgets.text("experiment_name", "/Shared/census_decision_tree_model")

source_table = dbutils.widgets.get("source_table")
experiment_name = dbutils.widgets.get("experiment_name")

mlflow.set_experiment(experiment_name)

print("Source table:", source_table)
print("MLflow experiment:", experiment_name)

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

categorical_features = [
    "workclass",
    "education_level",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
    "random_flag",
    "source_system",
]

model_df = df.select([label_col] + numeric_features + categorical_features)
model_df = model_df.filter(F.col(label_col).isin("<=80k", ">80k"))

for column_name in categorical_features:
    model_df = model_df.withColumn(column_name, F.coalesce(F.col(column_name), F.lit("Unknown")))

display(model_df.groupBy(label_col).count())
print("Model rows:", model_df.count())

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

one_hot_encoder = OneHotEncoder(
    inputCols=[f"{c}_idx" for c in categorical_features],
    outputCols=[f"{c}_ohe" for c in categorical_features],
    handleInvalid="keep",
)

numeric_assembler = VectorAssembler(
    inputCols=[f"{c}_imputed" for c in numeric_features],
    outputCol="numeric_features",
)

numeric_scaler = StandardScaler(
    inputCol="numeric_features",
    outputCol="scaled_numeric_features",
    withMean=False,
    withStd=True,
)

feature_assembler = VectorAssembler(
    inputCols=["scaled_numeric_features"] + [f"{c}_ohe" for c in categorical_features],
    outputCol="features",
)

decision_tree = DecisionTreeClassifier(
    featuresCol="features",
    labelCol="label",
    maxDepth=6,
    minInstancesPerNode=10,
    seed=42,
)

pipeline = Pipeline(
    stages=[
        label_indexer,
        numeric_imputer,
        *categorical_indexers,
        one_hot_encoder,
        numeric_assembler,
        numeric_scaler,
        feature_assembler,
        decision_tree,
    ]
)

# COMMAND ----------

auc_evaluator = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
)
accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy"
)
precision_evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="weightedPrecision"
)
recall_evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="weightedRecall"
)
f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="f1"
)


def evaluate(predictions):
    return {
        "accuracy": accuracy_evaluator.evaluate(predictions),
        "precision": precision_evaluator.evaluate(predictions),
        "recall": recall_evaluator.evaluate(predictions),
        "f1_score": f1_evaluator.evaluate(predictions),
        "roc_auc": auc_evaluator.evaluate(predictions),
    }


def get_feature_names(predictions, features_col="features"):
    attrs = predictions.schema[features_col].metadata.get("ml_attr", {}).get("attrs", {})
    feature_attrs = attrs.get("numeric", []) + attrs.get("binary", [])
    return [item["name"] for item in sorted(feature_attrs, key=lambda x: x["idx"])]

# COMMAND ----------

with mlflow.start_run(run_name="decision_tree") as run:
    mlflow.log_param("source_table", source_table)
    mlflow.log_param("model_type", "DecisionTreeClassifier")
    mlflow.log_param("maxDepth", 6)
    mlflow.log_param("minInstancesPerNode", 10)
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
    mlflow.spark.log_model(model, artifact_path="model")

print("Train metrics:", train_metrics)
print("Test metrics:", test_metrics)

# COMMAND ----------

print("Confusion matrix")
display(test_predictions.groupBy("label", "prediction").count().orderBy("label", "prediction"))

tree_model = model.stages[-1]
feature_names = get_feature_names(test_predictions)
importance_rows = list(zip(feature_names, tree_model.featureImportances.toArray().tolist()))
importance_df = spark.createDataFrame(importance_rows, ["feature", "importance"]).orderBy(F.desc("importance"))

display(importance_df.limit(30))
