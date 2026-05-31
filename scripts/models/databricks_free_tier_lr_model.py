# Databricks notebook source
# Free-tier friendly Logistic Regression for census income prediction.
# Designed to keep memory low: row limit, compact categorical encoding, small feature vector.

from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, Imputer, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import mlflow
from datetime import datetime

# COMMAND ----------

dbutils.widgets.text("source_table", "workshop.default.donation_data_v1")
dbutils.widgets.text("experiment_name", "/Shared/census_free_tier_lr")
dbutils.widgets.text("row_limit", "8000")
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

# Keep fewer categorical columns for free-tier memory safety.
categorical_features = [
    "workclass",
    "education_level",
    "marital_status",
    "occupation",
    "relationship",
    "race",
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

numeric_assembler = VectorAssembler(
    inputCols=[f"{c}_imputed" for c in numeric_features],
    outputCol="numeric_vector",
)

numeric_scaler = StandardScaler(
    inputCol="numeric_vector",
    outputCol="scaled_numeric_vector",
    withMean=False,
    withStd=True,
)

feature_columns = ["scaled_numeric_vector"] + [f"{c}_idx" for c in categorical_features]
feature_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=20,
    regParam=0.1,
    elasticNetParam=0.0,
)

pipeline = Pipeline(
    stages=[
        label_indexer,
        numeric_imputer,
        *categorical_indexers,
        numeric_assembler,
        numeric_scaler,
        feature_assembler,
        lr,
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

with mlflow.start_run(run_name="free_tier_logistic_regression"):
    mlflow.log_param("source_table", source_table)
    mlflow.log_param("row_limit", row_limit)
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("encoding", "StringIndexer compact encoding")
    mlflow.log_param("maxIter", 20)
    mlflow.log_param("regParam", 0.1)
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
