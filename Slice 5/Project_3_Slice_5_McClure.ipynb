{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "79931b7f-ec44-428d-8037-8f9b17579d3a",
   "metadata": {},
   "source": [
    "554 Project 3\n",
    "\n",
    "by Joshua McClure"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1d78a8fa-aaf3-46af-8da1-6aeb11de459b",
   "metadata": {},
   "source": [
    "Fitting Your Model (50 pts)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "a44bd3fe-ffa4-45e6-9d7d-7e1adf681d6b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Library for Packages\n",
    "import pandas as pd\n",
    "import time\n",
    "import os\n",
    "from pyspark.ml import Pipeline\n",
    "from pyspark.sql import SparkSession\n",
    "from pyspark.sql.types import StructType, StructField, DoubleType, StringType, IntegerType\n",
    "from pyspark.ml.feature import SQLTransformer, Binarizer, StringIndexer, OneHotEncoder, VectorAssembler, PCA\n",
    "from pyspark.ml.regression import LinearRegression\n",
    "from pyspark.ml.tuning import ParamGridBuilder, CrossValidator\n",
    "from pyspark.ml.evaluation import RegressionEvaluator\n",
    "from pyspark.sql.functions import col"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c8e256c3-44d4-48fb-aff8-dda2d9f2854c",
   "metadata": {},
   "source": [
    "Part 1: Read in and Organize Data\n",
    "\n",
    "Create a Jupyter notebook for the modeling fitting part and the Streaming part below.\n",
    "\n",
    "* The file power_ml_data.csv is available at the URL: https://www4.stat.ncsu.edu/~online/datasets/ power_ml_data.csv\n",
    "\n",
    "* You should read this data into a standard pandas data frame using the pd.read_csv() function.\n",
    "\n",
    "* Convert this to a spark data frame\n",
    "\n",
    "* We are going to treat the Power_Zone_3 variable as our response variable.\n",
    "\n",
    "* We can use all of the other variables as predictors. (Imagine we know that the Power_Zone_3 reading is going to go offline in the future and we need to be able to predict that value appropriately.)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "d9f7a0de-bca4-4293-a504-c684f78cef6a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Spark DataFrame Schema:\n",
      "root\n",
      " |-- Temperature: double (nullable = true)\n",
      " |-- Humidity: double (nullable = true)\n",
      " |-- Wind_Speed: double (nullable = true)\n",
      " |-- General_Diffuse_Flows: double (nullable = true)\n",
      " |-- Diffuse_Flows: double (nullable = true)\n",
      " |-- Power_Zone_1: double (nullable = true)\n",
      " |-- Power_Zone_2: double (nullable = true)\n",
      " |-- label: double (nullable = true)\n",
      " |-- Month: integer (nullable = true)\n",
      " |-- Hour: integer (nullable = true)\n",
      "\n",
      "\n",
      "First 5 rows of Spark DataFrame:\n",
      "+-----------+--------+----------+---------------------+-------------+------------+------------+-----------+-----+----+\n",
      "|Temperature|Humidity|Wind_Speed|General_Diffuse_Flows|Diffuse_Flows|Power_Zone_1|Power_Zone_2|      label|Month|Hour|\n",
      "+-----------+--------+----------+---------------------+-------------+------------+------------+-----------+-----+----+\n",
      "|      6.559|    73.8|     0.083|                0.051|        0.119|  34055.6962| 16128.87538|20240.96386|    1|   0|\n",
      "|      6.414|    74.5|     0.083|                 0.07|        0.085| 29814.68354| 19375.07599|20131.08434|    1|   0|\n",
      "|      6.313|    74.5|      0.08|                0.062|          0.1| 29128.10127| 19006.68693|19668.43373|    1|   0|\n",
      "|      6.121|    75.0|     0.083|                0.091|        0.096| 28228.86076| 18361.09422|18899.27711|    1|   0|\n",
      "|      5.921|    75.7|     0.081|                0.048|        0.085|  27335.6962| 17872.34043|18442.40964|    1|   0|\n",
      "+-----------+--------+----------+---------------------+-------------+------------+------------+-----------+-----+----+\n",
      "only showing top 5 rows\n"
     ]
    }
   ],
   "source": [
    "# Initialize SparkSession\n",
    "spark = SparkSession.builder.appName(\"PowerDataProcessing\").getOrCreate()\n",
    "\n",
    "# Define the URL for the dataset\n",
    "data_url = \"https://www4.stat.ncsu.edu/~online/datasets/power_ml_data.csv\"\n",
    "\n",
    "# Read the data into a pandas DataFrame, using the first row as headers\n",
    "pd_df = pd.read_csv(data_url, header=0)\n",
    "\n",
    "# Define the Spark schema based on the provided headers\n",
    "spark_schema = StructType([\n",
    "    StructField(\"Temperature\", DoubleType(), True),\n",
    "    StructField(\"Humidity\", DoubleType(), True),\n",
    "    StructField(\"Wind_Speed\", DoubleType(), True),\n",
    "    StructField(\"General_Diffuse_Flows\", DoubleType(), True),\n",
    "    StructField(\"Diffuse_Flows\", DoubleType(), True),\n",
    "    StructField(\"Power_Zone_1\", DoubleType(), True),\n",
    "    StructField(\"Power_Zone_2\", DoubleType(), True),\n",
    "    StructField(\"Power_Zone_3\", DoubleType(), True),\n",
    "    StructField(\"Month\", IntegerType(), True),\n",
    "    StructField(\"Hour\", IntegerType(), True)\n",
    "])\n",
    "\n",
    "# Convert pandas DataFrame to Spark DataFrame\n",
    "spark_df = spark.createDataFrame(pd_df, schema=spark_schema)\n",
    "\n",
    "# Rename Power_Zone_3 to 'label' as it is our response variable\n",
    "spark_df = spark_df.withColumnRenamed(\"Power_Zone_3\", \"label\")\n",
    "\n",
    "# Display schema and first few rows to verify\n",
    "print(\"Spark DataFrame Schema:\")\n",
    "spark_df.printSchema()\n",
    "\n",
    "print(\"\\nFirst 5 rows of Spark DataFrame:\")\n",
    "spark_df.show(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "78bb68d0-4069-4d2d-ad2a-1116e74add3a",
   "metadata": {},
   "source": [
    "Part 2: Elastic Model Initial Pipeline Set Up\n",
    "\n",
    "We want to fit an elastic net model using CV (no training/test split, just using CV on the data we’ve read\n",
    "in) with the steps below.\n",
    "\n",
    "The transformations below should each use an MLlib function that can be put into a pipeline\n",
    "\n",
    "  * The Hour column is likely not stored as a DoubleType. If it is not, use an SQL transformer to cast the variable as a DoubleType\n",
    "\n",
    "  * Binarize the Hour column based on the column being less than 6.5 or not (night vs day essentially)\n",
    "\n",
    "  * One-hot encode the Month column\n",
    "\n",
    "  * Run a PCA fit on the Temperature, Humidity, Wind_Speed, General_Diffuse_Flows, and Diffuse_Flows columns.\n",
    "\n",
    "    * To do this, I first used a VectorAssembler() call to place these variables in a column together for use with the PCA() estimator.\n",
    "\n",
    "    * Once fitted, then you’ll have a PCA transformer we’ll use in our pipeline.\n",
    "\n",
    "    * We’ll use two PCs in our transformation.\n",
    "\n",
    "* Rename your response variable as label\n",
    "\n",
    "* Use VectorAssembler() to put your predictors into a features. Use the:\n",
    "\n",
    "  * two fitted PCA features\n",
    "\n",
    "  * binary Hour variable\n",
    "\n",
    "  * Power_Zone_1\n",
    "\n",
    "  * Power_Zone_2\n",
    "\n",
    "  * Month indicator variables\n",
    "\n",
    "* This ends the pipeline of transformations!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "231308b2-9c95-4a12-97ec-567d794e3f33",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Pipeline stages defined successfully.\n"
     ]
    }
   ],
   "source": [
    "# Step 1: Cast Hour column to DoubleType if not already\n",
    "sql_transformer_hour = SQLTransformer(statement=\"SELECT *, CAST(Hour AS DOUBLE) AS Hour_Double FROM __THIS__\")\n",
    "\n",
    "# Step 2: Binarize the Hour column (Night vs Day)\n",
    "binarizer = Binarizer(threshold=6.5, inputCol=\"Hour_Double\", outputCol=\"Hour_Binary\")\n",
    "\n",
    "# Step 3: One-hot encode the Month column\n",
    "month_indexer = StringIndexer(inputCol=\"Month\", outputCol=\"Month_Indexed\")\n",
    "\n",
    "# Then, OneHotEncoder to convert the indexed column to one-hot vectors\n",
    "month_encoder = OneHotEncoder(inputCols=[\"Month_Indexed\"], outputCols=[\"Month_OneHot\"])\n",
    "\n",
    "# Step 4: PCA on Temperature, Humidity, Wind_Speed, General_Diffuse_Flows, and Diffuse_Flows\n",
    "pca_input_cols = [\"Temperature\", \"Humidity\", \"Wind_Speed\", \"General_Diffuse_Flows\", \"Diffuse_Flows\"]\n",
    "vector_assembler_pca = VectorAssembler(inputCols=pca_input_cols, outputCol=\"pca_features\")\n",
    "\n",
    "# Apply PCA to reduce dimensions to 2 principal components\n",
    "pca = PCA(k=2, inputCol=\"pca_features\", outputCol=\"principal_components\")\n",
    "\n",
    "# Step 5: Final VectorAssembler to put all predictors into a 'features' vector\n",
    "# Use the two fitted PCA features, binary Hour, Power_Zone_1, Power_Zone_2, and Month indicator variables\n",
    "final_features_assembler = VectorAssembler(\n",
    "    inputCols=[\n",
    "        \"principal_components\",\n",
    "        \"Hour_Binary\",\n",
    "        \"Power_Zone_1\",\n",
    "        \"Power_Zone_2\",\n",
    "        \"Month_OneHot\"\n",
    "    ],\n",
    "    outputCol=\"features\"\n",
    ")\n",
    "\n",
    "# Step 6: Define the Linear Regression model\n",
    "lr = LinearRegression(featuresCol=\"features\", labelCol=\"label\", predictionCol=\"prediction\")\n",
    "\n",
    "# Create the pipeline\n",
    "pipeline = Pipeline(stages=[\n",
    "    sql_transformer_hour,\n",
    "    binarizer,\n",
    "    month_indexer,\n",
    "    month_encoder,\n",
    "    vector_assembler_pca,\n",
    "    pca,\n",
    "    final_features_assembler,\n",
    "    lr # Add the Linear Regression model as the final stage of the pipeline\n",
    "])\n",
    "\n",
    "print(\"Pipeline stages defined successfully.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "faa46c36-0675-4dc8-88d1-e185266279ea",
   "metadata": {},
   "source": [
    "Part 3: Fitting & Printing the Model\n",
    "\n",
    "* Now you’ll then use the CrossValidator() function and the LinearRegression() function to fit an elastic net model.\n",
    "\n",
    "  * You should do the following grid for the regParam and elasticNetParam: All combinations of\n",
    "\n",
    "    * regParam: 0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 1\n",
    "\n",
    "    * elasticNetParam: 0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 1\n",
    "\n",
    "* Now fit the model using 5-fold CV with rmse as your criterion!\n",
    "\n",
    "* Report the optimal values chosen for the tuning parameters\n",
    "\n",
    "* Report the CV error\n",
    "\n",
    "* Report the training set RMSE (as done in the notes) by using your fitted model as a transformer and evaluating on the entire training set\n",
    "\n",
    "* Take the outputted transformations from the model (the predictions) and create a residual column (label - prediction). The .withColumn() method is handy here. Print out a data frame with these residuals, the label column, and the predictions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "369b273e-ecb3-4008-a709-ad426a9bb443",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting the model using CrossValidator...\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "26/04/30 12:29:25 WARN SparkStringUtils: Truncated the string representation of a plan since it was too large. This behavior can be adjusted by setting 'spark.sql.debug.maxToStringFields'.\n",
      "26/04/30 12:29:26 WARN Instrumentation: [6d27cfb4] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:29:29 WARN Instrumentation: [0289fe96] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:29:31 WARN Instrumentation: [673e76de] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:29:33 WARN Instrumentation: [ea633f71] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:29:34 WARN Instrumentation: [ff262512] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:29:35 WARN Instrumentation: [504962c7] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:29:36 WARN Instrumentation: [2d4795d1] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:29:38 WARN Instrumentation: [975709b9] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:29:39 WARN Instrumentation: [c95c99f2] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:29:40 WARN Instrumentation: [1ff60e19] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:29:41 WARN Instrumentation: [743fafe3] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:32:21 WARN Instrumentation: [14ced0fb] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:32:23 WARN Instrumentation: [1555c5a3] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:32:24 WARN Instrumentation: [04139fe8] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:32:25 WARN Instrumentation: [a974fea6] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:32:26 WARN Instrumentation: [9c229850] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:32:27 WARN Instrumentation: [06762453] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:32:28 WARN Instrumentation: [e22d5a0e] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:32:29 WARN Instrumentation: [0f9498a5] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:32:30 WARN Instrumentation: [b6eb9816] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:32:31 WARN Instrumentation: [f5a55456] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:32:32 WARN Instrumentation: [82b15fd9] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:34:23 WARN Instrumentation: [418ec6f1] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:34:24 WARN Instrumentation: [f79f8d80] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:34:25 WARN Instrumentation: [a18ad02d] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:34:26 WARN Instrumentation: [353b5eb8] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:34:27 WARN Instrumentation: [a1a3e544] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:34:28 WARN Instrumentation: [0121ee04] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:34:29 WARN Instrumentation: [5cbd636e] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:34:30 WARN Instrumentation: [06f19d98] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:34:31 WARN Instrumentation: [1405444f] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:34:32 WARN Instrumentation: [e08ff2a3] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:34:33 WARN Instrumentation: [dffe3bce] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:36:23 WARN Instrumentation: [b1c64e6b] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:36:24 WARN Instrumentation: [763a778b] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:36:25 WARN Instrumentation: [d6110f7a] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:36:26 WARN Instrumentation: [be04d029] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:36:27 WARN Instrumentation: [138cda97] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:36:28 WARN Instrumentation: [e44beb7a] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:36:29 WARN Instrumentation: [245954d5] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:36:30 WARN Instrumentation: [77d851d1] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:36:31 WARN Instrumentation: [380fac94] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:36:32 WARN Instrumentation: [f2f4abb5] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:36:33 WARN Instrumentation: [0a31a296] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:38:34 WARN Instrumentation: [04dbc0c0] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:38:36 WARN Instrumentation: [07134790] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:38:37 WARN Instrumentation: [43afe70d] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:38:38 WARN Instrumentation: [83372193] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:38:39 WARN Instrumentation: [08868435] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:38:40 WARN Instrumentation: [b01e2cef] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:38:41 WARN Instrumentation: [bd4c7448] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:38:42 WARN Instrumentation: [02fab67b] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:38:43 WARN Instrumentation: [92dc8953] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:38:44 WARN Instrumentation: [ec2d9780] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 12:38:45 WARN Instrumentation: [57d653c5] regParam is zero, which might cause numerical instability and overfitting.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model fitting complete.\n",
      "Could not find LinearRegression stage in the best model.\n",
      "\n",
      "Cross-validation RMSE (best model): 2147.5891325226876\n",
      "Training set RMSE: 2147.097322400667\n",
      "\n",
      "DataFrame with Label, Prediction, and Residuals:\n",
      "+-----------+------------------+------------------+\n",
      "|      label|        prediction|          residual|\n",
      "+-----------+------------------+------------------+\n",
      "|20240.96386|20879.293929772837|-638.3300697728373|\n",
      "|20131.08434|18659.581224684986|1471.5031153150157|\n",
      "|19668.43373|18204.118378530166|1464.3153514698352|\n",
      "|18899.27711|17590.065188636432| 1309.211921363567|\n",
      "|18442.40964|16996.736644024797|1445.6729959752047|\n",
      "|18130.12048| 16517.14980719943| 1612.970672800573|\n",
      "|17945.06024|16092.738824696906| 1852.321415303093|\n",
      "|17459.27711|15722.205354351358|1737.0717556486416|\n",
      "|17025.54217| 15270.58168727468|1754.9604827253206|\n",
      "|16794.21687|14937.899896525745| 1856.316973474255|\n",
      "+-----------+------------------+------------------+\n",
      "only showing top 10 rows\n"
     ]
    }
   ],
   "source": [
    "# The Linear Regression model is already part of the pipeline defined in the previous cell.\n",
    "# We need to retrieve that instance to correctly define the ParamGridBuilder.\n",
    "lr_from_pipeline = None\n",
    "for stage in pipeline.getStages():\n",
    "    if isinstance(stage, LinearRegression):\n",
    "        lr_from_pipeline = stage\n",
    "        break\n",
    "\n",
    "if lr_from_pipeline is None:\n",
    "    raise ValueError(\"LinearRegression stage not found in the pipeline. Please ensure the pipeline in the previous cell is correctly defined and executed with a LinearRegression model.\")\n",
    "\n",
    "# Define the parameter grid for regParam and elasticNetParam\n",
    "paramGrid = ParamGridBuilder() \\\n",
    "    .addGrid(lr_from_pipeline.regParam, [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 1.0]) \\\n",
    "    .addGrid(lr_from_pipeline.elasticNetParam, [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 1.0]) \\\n",
    "    .build()\n",
    "\n",
    "# Create a RegressionEvaluator for RMSE\n",
    "evaluator = RegressionEvaluator(labelCol=\"label\", predictionCol=\"prediction\", metricName=\"rmse\")\n",
    "\n",
    "# Create the CrossValidator\n",
    "cv = CrossValidator(\n",
    "    estimator=pipeline, # Use the previously defined pipeline as the estimator\n",
    "    estimatorParamMaps=paramGrid,\n",
    "    evaluator=evaluator,\n",
    "    numFolds=5, \n",
    "    seed=42 \n",
    ")\n",
    "\n",
    "print(\"Fitting the model using CrossValidator...\")\n",
    "# Fit the model to the spark_df\n",
    "cvModel = cv.fit(spark_df)\n",
    "\n",
    "print(\"Model fitting complete.\")\n",
    "\n",
    "# --- Reporting Results ---\n",
    "\n",
    "# The best model from CrossValidator is stored in cvModel.bestModel\n",
    "best_pipeline_model = cvModel.bestModel\n",
    "\n",
    "# Find the LinearRegressionModel stage in the best pipeline model\n",
    "best_lr_model = None\n",
    "for stage in best_pipeline_model.stages:\n",
    "    if isinstance(stage, LinearRegression):\n",
    "        best_lr_model = stage\n",
    "        break\n",
    "\n",
    "if best_lr_model:\n",
    "    print(f\"\\nOptimal regParam: {best_lr_model.getRegParam()}\")\n",
    "    print(f\"Optimal elasticNetParam: {best_lr_model.getElasticNetParam()}\")\n",
    "else:\n",
    "    print(\"Could not find LinearRegression stage in the best model.\")\n",
    "\n",
    "# Report the CV error (average RMSE from cross-validation)\n",
    "best_rmse = min(cvModel.avgMetrics)\n",
    "print(f\"\\nCross-validation RMSE (best model): {best_rmse}\")\n",
    "\n",
    "# Report the training set RMSE\n",
    "transformed_df = cvModel.transform(spark_df)\n",
    "\n",
    "training_rmse = evaluator.evaluate(transformed_df)\n",
    "print(f\"Training set RMSE: {training_rmse}\")\n",
    "\n",
    "# Take the outputted transformations (predictions) and create a residual column\n",
    "residuals_df = transformed_df.withColumn(\"residual\", col(\"label\") - col(\"prediction\"))\n",
    "\n",
    "print(\"\\nDataFrame with Label, Prediction, and Residuals:\")\n",
    "residuals_df.select(\"label\", \"prediction\", \"residual\").show(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f20bcb2f-3b41-4943-99ce-c38c558f7059",
   "metadata": {},
   "source": [
    "Streaming Part (40 pts)\n",
    "\n",
    "There is another file available at: https://www4.stat.ncsu.edu/~online/datasets/power_streaming_data.csv\n",
    "Download this file and store it where your .py file you’ll create can find it. We’ll be randomly sampling rows\n",
    "from this to output to .csv files that you’ll be reading in."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "c13a8093-8818-4772-926b-cec58bab5b5d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Streaming DataFrame Schema:\n",
      "root\n",
      " |-- Temperature: double (nullable = true)\n",
      " |-- Humidity: double (nullable = true)\n",
      " |-- Wind_Speed: double (nullable = true)\n",
      " |-- General_Diffuse_Flows: double (nullable = true)\n",
      " |-- Diffuse_Flows: double (nullable = true)\n",
      " |-- Power_Zone_1: double (nullable = true)\n",
      " |-- Power_Zone_2: double (nullable = true)\n",
      " |-- Power_Zone_3: double (nullable = true)\n",
      " |-- Month: integer (nullable = true)\n",
      " |-- Hour: integer (nullable = true)\n",
      "\n",
      "First 5 rows of Streaming DataFrame:\n",
      "+-----------+--------+----------+---------------------+-------------+------------+------------+------------+-----+----+\n",
      "|Temperature|Humidity|Wind_Speed|General_Diffuse_Flows|Diffuse_Flows|Power_Zone_1|Power_Zone_2|Power_Zone_3|Month|Hour|\n",
      "+-----------+--------+----------+---------------------+-------------+------------+------------+------------+-----+----+\n",
      "|      4.805|    76.2|     0.081|                0.059|        0.134| 20421.26582| 12908.20669| 14590.84337|    1|   3|\n",
      "|      4.212|    78.3|     0.081|                0.117|        0.082| 21393.41772| 13575.68389|  14862.6506|    1|   5|\n",
      "|      4.304|    76.0|     0.082|                0.048|        0.152| 19983.79747| 12342.85714| 13492.04819|    1|   7|\n",
      "|      4.489|    74.3|     0.082|                0.081|        0.119| 18167.08861| 11551.36778| 11600.96386|    1|   7|\n",
      "|      4.509|    74.5|     0.084|                6.643|        6.494| 19837.97468| 11945.28875| 11178.79518|    1|   8|\n",
      "+-----------+--------+----------+---------------------+-------------+------------+------------+------------+-----+----+\n",
      "only showing top 5 rows\n"
     ]
    }
   ],
   "source": [
    "# Define the schema for the streaming data\n",
    "spark_schema_streaming = StructType([\n",
    "    StructField(\"Temperature\", DoubleType(), True),\n",
    "    StructField(\"Humidity\", DoubleType(), True),\n",
    "    StructField(\"Wind_Speed\", DoubleType(), True),\n",
    "    StructField(\"General_Diffuse_Flows\", DoubleType(), True),\n",
    "    StructField(\"Diffuse_Flows\", DoubleType(), True),\n",
    "    StructField(\"Power_Zone_1\", DoubleType(), True),\n",
    "    StructField(\"Power_Zone_2\", DoubleType(), True),\n",
    "    StructField(\"Power_Zone_3\", DoubleType(), True),\n",
    "    StructField(\"Month\", IntegerType(), True),\n",
    "    StructField(\"Hour\", IntegerType(), True)\n",
    "])\n",
    "\n",
    "# Define the path to the streaming data file\n",
    "streaming_data_path = \"Power_Storage/power_streaming_data.csv\"\n",
    "\n",
    "# Read the data into a Spark DataFrame\n",
    "streaming_df = spark.read.csv(\n",
    "    streaming_data_path,\n",
    "    header=True,\n",
    "    schema=spark_schema_streaming\n",
    ")\n",
    "\n",
    "print(\"Streaming DataFrame Schema:\")\n",
    "streaming_df.printSchema()\n",
    "\n",
    "print(\"First 5 rows of Streaming DataFrame:\")\n",
    "streaming_df.show(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2b6ca367-9802-4ab6-9bcc-26ec8dbf15c0",
   "metadata": {},
   "source": [
    "Part 1: Reading a Stream\n",
    "\n",
    "* We’re going to read in a stream in the form of .csv files. Create a folder where you will be sending your .csv files.\n",
    "\n",
    "* Setup the schema for the stream (you can use the schema from the original data as we did in hw 10)\n",
    "\n",
    "* Set up the readStream. Be sure to add header = True as you’ll likely be outputting files with a header and we don’t need to read that in."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "1cf6b8eb-1359-442f-953d-5450b57d2d86",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Streaming input directory already exists: Power_Storage/stream_data_input\n",
      "Spark Structured Stream configured to read from:\n",
      "  Directory: Power_Storage/stream_data_input\n",
      "root\n",
      " |-- Temperature: double (nullable = true)\n",
      " |-- Humidity: double (nullable = true)\n",
      " |-- Wind_Speed: double (nullable = true)\n",
      " |-- General_Diffuse_Flows: double (nullable = true)\n",
      " |-- Diffuse_Flows: double (nullable = true)\n",
      " |-- Power_Zone_1: double (nullable = true)\n",
      " |-- Power_Zone_2: double (nullable = true)\n",
      " |-- Power_Zone_3: double (nullable = true)\n",
      " |-- Month: integer (nullable = true)\n",
      " |-- Hour: integer (nullable = true)\n",
      "\n",
      "  Schema: None\n",
      "Stream initialized successfully.\n"
     ]
    }
   ],
   "source": [
    "# 1. Create a folder where you will be sending your .csv files\n",
    "#    This will be the input directory for our Spark Structured Stream\n",
    "streaming_input_dir = \"Power_Storage/stream_data_input\"\n",
    "if not os.path.exists(streaming_input_dir):\n",
    "    os.makedirs(streaming_input_dir)\n",
    "    print(f\"Created streaming input directory: {streaming_input_dir}\")\n",
    "else:\n",
    "    print(f\"Streaming input directory already exists: {streaming_input_dir}\")\n",
    "\n",
    "# 2. Set up the readStream\n",
    "# We'll monitor the streaming_input_dir for new CSV files\n",
    "stream_df = spark.readStream \\\n",
    "    .format(\"csv\") \\\n",
    "    .option(\"header\", \"true\") \\\n",
    "    .schema(spark_schema_streaming) \\\n",
    "    .load(streaming_input_dir)\n",
    "\n",
    "print(\"Spark Structured Stream configured to read from:\")\n",
    "print(f\"  Directory: {streaming_input_dir}\")\n",
    "print(f\"  Schema: {stream_df.printSchema()}\")\n",
    "print(\"Stream initialized successfully.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c19a1192-5df5-439a-9c9f-c1c0516fd562",
   "metadata": {},
   "source": [
    "Part 2: Transform/Aggregation Step\n",
    "\n",
    "* Now, we’ll do two separate things on the stream and join them together:\n",
    "\n",
    "  * With your stream, use your model transformer to obtain predictions from the incoming data. On the resulting predictions also create a residual column as noted in the previous section (return only the label, prediction and residual columns from this part)\n",
    "\n",
    "  * We can use our stream more than once! With another transformation on the (original) stream, modify the response variable to be called label.\n",
    "\n",
    "  * Now join your above transform with this stream based on the label variable which should be common to both!\n",
    "\n",
    "  * Note 1: This is a little silly, but I want you to join two transformations of the stream and I don’t want things to get too crazy\n",
    "\n",
    "  * Note 2: Each data frame is created from the same stream of data! You don’t need two streams, you can use the same stream and just do two separate transformations on it, combining it with a .join() method from one of the SQL style data frames you are dealing with (as we discussed in the notes)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "ad414b56-7398-4f32-9d14-cc8d157efe8d",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "26/04/30 13:01:48 WARN SparkStringUtils: Truncated the string representation of a plan since it was too large. This behavior can be adjusted by setting 'spark.sql.debug.maxToStringFields'.\n",
      "26/04/30 13:01:48 WARN Instrumentation: [20f4cfff] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:01:52 WARN Instrumentation: [6b90dafd] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:01:53 WARN Instrumentation: [cead3908] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:01:55 WARN Instrumentation: [e94d77db] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:01:56 WARN Instrumentation: [4b603d70] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:01:57 WARN Instrumentation: [7891ed91] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:01:59 WARN Instrumentation: [7368e2c7] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:02:00 WARN Instrumentation: [a06ad86e] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:02:01 WARN Instrumentation: [2cf73649] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:02:02 WARN Instrumentation: [99907dea] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:02:03 WARN Instrumentation: [a17f86b6] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:04:01 WARN Instrumentation: [b9b909a2] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:04:02 WARN Instrumentation: [7e54507e] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:04:03 WARN Instrumentation: [9d5fda80] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:04:04 WARN Instrumentation: [22373eb3] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:04:05 WARN Instrumentation: [3068ca95] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:04:06 WARN Instrumentation: [dfdd347b] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:04:07 WARN Instrumentation: [8fada6f8] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:04:08 WARN Instrumentation: [6a3f39a9] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:04:09 WARN Instrumentation: [04c93165] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:04:10 WARN Instrumentation: [3cbb602d] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:04:11 WARN Instrumentation: [64fd0e85] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:06:04 WARN Instrumentation: [7ec2e007] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:06:05 WARN Instrumentation: [01ec9a16] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:06:06 WARN Instrumentation: [278083ae] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:06:07 WARN Instrumentation: [45ed8dab] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:06:08 WARN Instrumentation: [13e05348] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:06:09 WARN Instrumentation: [bc73c525] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:06:10 WARN Instrumentation: [01e2cc4e] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:06:11 WARN Instrumentation: [bf738fd5] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:06:12 WARN Instrumentation: [9fa1794c] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:06:13 WARN Instrumentation: [a39aff5f] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:06:14 WARN Instrumentation: [c745d2b7] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:08:07 WARN Instrumentation: [e129cdd9] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:08:08 WARN Instrumentation: [0745eab7] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:08:09 WARN Instrumentation: [df8e75f8] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:08:10 WARN Instrumentation: [661f7def] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:08:11 WARN Instrumentation: [b2f391e4] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:08:12 WARN Instrumentation: [70813e2c] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:08:13 WARN Instrumentation: [26a63845] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:08:14 WARN Instrumentation: [70e17a40] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:08:15 WARN Instrumentation: [489da835] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:08:16 WARN Instrumentation: [814616ea] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:08:17 WARN Instrumentation: [128192e9] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:10:09 WARN Instrumentation: [700637fa] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:10:10 WARN Instrumentation: [466a88d5] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:10:11 WARN Instrumentation: [c1e22b92] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:10:12 WARN Instrumentation: [967a28d0] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:10:13 WARN Instrumentation: [c1773c8d] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:10:14 WARN Instrumentation: [bbe461fb] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:10:15 WARN Instrumentation: [3eadff30] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:10:16 WARN Instrumentation: [fa7e1052] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:10:17 WARN Instrumentation: [56887556] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:10:18 WARN Instrumentation: [9cd3d205] regParam is zero, which might cause numerical instability and overfitting.\n",
      "26/04/30 13:10:19 WARN Instrumentation: [a924e2be] regParam is zero, which might cause numerical instability and overfitting.\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Streaming transformations and join pipeline defined.\n"
     ]
    }
   ],
   "source": [
    "# The Linear Regression model is already part of the pipeline defined in the previous cell.\n",
    "# We need to retrieve that instance to correctly define the ParamGridBuilder.\n",
    "lr_from_pipeline = None\n",
    "for stage in pipeline.getStages():\n",
    "    if isinstance(stage, LinearRegression):\n",
    "        lr_from_pipeline = stage\n",
    "        break\n",
    "\n",
    "if lr_from_pipeline is None:\n",
    "    raise ValueError(\"LinearRegression stage not found in the pipeline. Please ensure the pipeline in the previous cell is correctly defined and executed with a LinearRegression model.\")\n",
    "\n",
    "# Define the parameter grid for regParam and elasticNetParam\n",
    "paramGrid = ParamGridBuilder() \\\n",
    "    .addGrid(lr_from_pipeline.regParam, [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 1.0]) \\\n",
    "    .addGrid(lr_from_pipeline.elasticNetParam, [0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 1.0]) \\\n",
    "    .build()\n",
    "\n",
    "# Create a RegressionEvaluator for RMSE\n",
    "evaluator = RegressionEvaluator(labelCol=\"label\", predictionCol=\"prediction\", metricName=\"rmse\")\n",
    "\n",
    "# Create the CrossValidator\n",
    "cv = CrossValidator(\n",
    "    estimator=pipeline, # Use the previously defined pipeline as the estimator\n",
    "    estimatorParamMaps=paramGrid,\n",
    "    evaluator=evaluator,\n",
    "    numFolds=5, \n",
    "    seed=42 \n",
    ")\n",
    "\n",
    "# Adding in model again since original code block takes a long time to run\n",
    "cvModel = cv.fit(spark_df)\n",
    "\n",
    "#--------------------------------------------------------------------------------------------------#\n",
    "\n",
    "# 1. Prepare stream for model by renaming Power_Zone_3 to label\n",
    "stream_for_model = stream_df.withColumnRenamed(\"Power_Zone_3\", \"label\")\n",
    "\n",
    "# 2. Use the model transformer to obtain predictions from the incoming data and create a residual column.\n",
    "predicted_stream = cvModel.transform(stream_for_model)\n",
    "\n",
    "# Return only the label, prediction, and residual columns from this part\n",
    "predictions_residuals_stream = predicted_stream \\\n",
    "    .withColumn(\"residual\", col(\"label\") - col(\"prediction\")) \\\n",
    "    .select(\"label\", \"prediction\", \"residual\")\n",
    "\n",
    "# 3. With another transformation on the (original) stream, modify the response variable to be called label.\n",
    "#    This creates a second stream with the label for joining.\n",
    "original_stream_with_label = stream_df.withColumnRenamed(\"Power_Zone_3\", \"label\")\n",
    "\n",
    "# 4. Join your above transform (predictions_residuals_stream) with this stream\n",
    "\n",
    "joined_stream = predictions_residuals_stream.alias(\"pred\") \\\n",
    "    .join(original_stream_with_label.alias(\"orig\"), \n",
    "          col(\"pred.label\") == col(\"orig.label\"), \n",
    "          \"inner\") \\\n",
    "    .select(\"pred.label\", \"pred.prediction\", \"pred.residual\", col(\"orig.*\"))\n",
    "\n",
    "print(\"Streaming transformations and join pipeline defined.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e7fdd0b2-8c3f-45c7-9eee-0b253793deb8",
   "metadata": {},
   "source": [
    "Part 3: Writing Step\n",
    "\n",
    "* Now write your stream to the console using the append output mode.\n",
    "\n",
    "* Start the query!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "8decce99-596b-45a3-a46d-0ed4a9c78038",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Streaming query started. Output will appear below.\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "26/04/30 13:12:53 WARN ResolveWriteToStream: Temporary checkpoint location created which is deleted normally when the query didn't fail: /tmp/temporary-1f38643a-2b5b-4508-bbfa-d522bf6fb6ff. If it's required to delete it under any circumstances, please set spark.sql.streaming.forceDeleteTempCheckpointLocation to true. Important to know deleting temp checkpoint folder is best effort.\n",
      "26/04/30 13:12:53 WARN ResolveWriteToStream: spark.sql.adaptive.enabled is not supported in streaming DataFrames/Datasets and will be disabled.\n"
     ]
    }
   ],
   "source": [
    "streamingQuery = joined_stream.writeStream \\\n",
    "    .outputMode(\"append\") \\\n",
    "    .format(\"console\") \\\n",
    "    .start()\n",
    "\n",
    "print(\"Streaming query started. Output will appear below.\")\n",
    "# To stop the query, you can run streamingQuery.stop() in a new cell."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a3f7d398-44d1-45ec-9053-a2989f9cc931",
   "metadata": {},
   "source": [
    "Produce Data (10 pts)\n",
    "\n",
    "You should have the file we’ll use for streaming data downloaded and in a place you can locate. Create a\n",
    ".py file that reads that into a pandas (regular) data frame and does the following.\n",
    "\n",
    "* Writes a loop (say 20 iterations) to:\n",
    "\n",
    "  * Randomly sample five rows and output those to a .csv file in the folder you are watching with your stream.\n",
    "\n",
    "  * Be sure not to write out the indices. You can leave the column names as long as you handle that on your stream appropriately\n",
    "\n",
    "  * Pause for 10 seconds in between outputting of data sets\n",
    "\n",
    "  * Submit this loop in a python console.\n",
    "\n",
    "* While the loop runs, you should see output in your notebook!"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "acb1103f-23fe-4e3e-b268-f074b2a39928",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Successfully loaded 5242 rows from Power_Storage/power_streaming_data.csv\n",
      "Starting streaming data simulation...\n",
      "Iteration 1/20: Wrote 5 rows to Power_Storage/stream_data_input/stream_data_part_00.csv\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "-------------------------------------------\n",
      "Batch: 0\n",
      "-------------------------------------------\n",
      "+-----------+------------------+-------------------+-----------+--------+----------+---------------------+-------------+------------+------------+-----------+-----+----+\n",
      "|      label|        prediction|           residual|Temperature|Humidity|Wind_Speed|General_Diffuse_Flows|Diffuse_Flows|Power_Zone_1|Power_Zone_2|      label|Month|Hour|\n",
      "+-----------+------------------+-------------------+-----------+--------+----------+---------------------+-------------+------------+------------+-----------+-----+----+\n",
      "|13208.58018|11463.393568338957| 1745.1866116610436|       20.9|    72.9|     4.916|                0.084|        0.104| 25595.04425|  14995.0104|13208.58018|    9|   4|\n",
      "|16037.41935| 16702.60799553148| -665.1886455314816|      17.81|    71.5|     0.083|                660.2|        52.58| 32403.06383|     22650.0|16037.41935|    3|  12|\n",
      "|16175.42169|18855.554333181317|-2680.1326431813177|      11.38|   66.54|      0.08|                13.07|        13.45|  32603.5443| 22460.79027|16175.42169|    1|  15|\n",
      "|14638.06452|13475.482626011559| 1162.5818939884412|      10.66|    88.9|     0.078|                0.081|        0.163| 23223.82979| 13178.04878|14638.06452|    3|   2|\n",
      "|25082.51046|24232.075511894098|  850.4349481059035|      21.54|   61.46|     4.908|                250.1|        230.9| 30994.28571| 22598.73418|25082.51046|    7|   8|\n",
      "+-----------+------------------+-------------------+-----------+--------+----------+---------------------+-------------+------------+------------+-----------+-----+----+\n",
      "\n",
      "Iteration 2/20: Wrote 5 rows to Power_Storage/stream_data_input/stream_data_part_01.csv\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[Stage 11525:=================================>                (132 + 68) / 200]\r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Iteration 3/20: Wrote 5 rows to Power_Storage/stream_data_input/stream_data_part_02.csv\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "-------------------------------------------\n",
      "Batch: 1\n",
      "-------------------------------------------\n",
      "+-----------+------------------+-------------------+-----------+--------+----------+---------------------+-------------+------------+------------+-----------+-----+----+\n",
      "|      label|        prediction|           residual|Temperature|Humidity|Wind_Speed|General_Diffuse_Flows|Diffuse_Flows|Power_Zone_1|Power_Zone_2|      label|Month|Hour|\n",
      "+-----------+------------------+-------------------+-----------+--------+----------+---------------------+-------------+------------+------------+-----------+-----+----+\n",
      "|24865.60669|27357.808775288315|-2492.2020852883143|      26.33|    73.4|     4.926|                691.5|        395.0| 38183.12292| 24505.06329|24865.60669|    7|  12|\n",
      "|14388.43373|13377.552169481965| 1010.8815605180353|      16.96|   60.88|     4.918|                0.055|        0.115| 22152.91139| 13415.19757|14388.43373|    1|   5|\n",
      "|22985.83072| 24189.62786898032|-1203.7971489803167|      26.72|   52.92|     4.908|                574.6|        41.28| 37583.84018| 24781.83738|22985.83072|    8|  16|\n",
      "|14485.31154|11881.151517547165| 2604.1600224528356|      21.14|   56.25|     4.917|                0.099|        0.048| 26181.23894| 15586.27859|14485.31154|    9|   4|\n",
      "|8418.727491| 9482.034161184118| -1063.306670184118|      11.96|    52.0|     0.076|                150.7|        34.56| 27668.44106|  22471.9239|8418.727491|   12|   9|\n",
      "+-----------+------------------+-------------------+-----------+--------+----------+---------------------+-------------+------------+------------+-----------+-----+----+\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[Stage 11528:================================>                 (128 + 72) / 200]\r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Iteration 4/20: Wrote 5 rows to Power_Storage/stream_data_input/stream_data_part_03.csv\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "-------------------------------------------\n",
      "Batch: 2\n",
      "-------------------------------------------\n",
      "+-----------+------------------+------------------+-----------+--------+----------+---------------------+-------------+------------+------------+-----------+-----+----+\n",
      "|      label|        prediction|          residual|Temperature|Humidity|Wind_Speed|General_Diffuse_Flows|Diffuse_Flows|Power_Zone_1|Power_Zone_2|      label|Month|Hour|\n",
      "+-----------+------------------+------------------+-----------+--------+----------+---------------------+-------------+------------+------------+-----------+-----+----+\n",
      "|10372.14886| 9371.635133602234|1000.5137263977649|      17.02|   63.08|     0.084|                462.1|        60.18|  28714.8289| 22552.93035|10372.14886|   12|  13|\n",
      "|22709.16923|16174.953569519115| 6534.215660480884|      27.01|   57.24|     4.918|                871.0|        106.7| 30917.08609| 18782.12058|22709.16923|    6|  14|\n",
      "|22943.59833|25780.737552059196|-2837.139222059195|       23.9|    75.0|     4.911|                0.106|        0.115| 29437.87375| 18189.87342|22943.59833|    7|   2|\n",
      "|15033.88945| 17305.22253178603|-2271.333081786028|      15.63|   40.52|     0.085|                587.2|        39.07| 32912.54237| 20564.13374|15033.88945|    2|  15|\n",
      "|25555.66265|25537.223388803464|18.439261196534062|      14.53|    74.1|     0.085|                0.051|        0.159| 42932.65823| 25798.17629|25555.66265|    1|  20|\n",
      "+-----------+------------------+------------------+-----------+--------+----------+---------------------+-------------+------------+------------+-----------+-----+----+\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[Stage 11531:===>                                              (14 + 128) / 200]\r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Iteration 5/20: Wrote 5 rows to Power_Storage/stream_data_input/stream_data_part_04.csv\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "-------------------------------------------\n",
      "Batch: 3\n",
      "-------------------------------------------\n",
      "+-----------+------------------+-------------------+-----------+--------+----------+---------------------+-------------+------------+------------+-----------+-----+----+\n",
      "|      label|        prediction|           residual|Temperature|Humidity|Wind_Speed|General_Diffuse_Flows|Diffuse_Flows|Power_Zone_1|Power_Zone_2|      label|Month|Hour|\n",
      "+-----------+------------------+-------------------+-----------+--------+----------+---------------------+-------------+------------+------------+-----------+-----+----+\n",
      "|34036.36364| 30506.20174394689| 3530.1618960531123|      26.02|   63.95|     4.905|                0.106|        0.078| 44392.27525| 32407.60296|34036.36364|    8|  22|\n",
      "|18626.95385|19895.635017384633|-1268.6811673846314|      24.73|   56.51|     0.071|                453.2|        168.5| 35272.05298| 22015.38462|18626.95385|    6|  16|\n",
      "|26204.51613| 24965.25886407466| 1239.2572659253383|      16.88|   68.68|     4.917|                3.348|        3.085|     43200.0| 24018.29268|26204.51613|    3|  19|\n",
      "|9991.836735|11404.442258007599| -1412.605523007598|      13.55|   59.93|     0.086|                499.4|        37.36| 31568.06084| 25539.12243|9991.836735|   12|  12|\n",
      "| 14845.3012|13857.702701698414|   987.598498301586|       8.23|    80.7|     0.076|                0.059|        0.089| 22772.65823|  14378.1155| 14845.3012|    1|   1|\n",
      "+-----------+------------------+-------------------+-----------+--------+----------+---------------------+-------------+------------+------------+-----------+-----+----+\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[Stage 11534:=>                                                 (6 + 128) / 200]\r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Iteration 6/20: Wrote 5 rows to Power_Storage/stream_data_input/stream_data_part_05.csv\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "-------------------------------------------\n",
      "Batch: 4\n",
      "-------------------------------------------\n",
      "+-----------+------------------+------------------+-----------+--------+----------+---------------------+-------------+------------+------------+-----------+-----+----+\n",
      "|      label|        prediction|          residual|Temperature|Humidity|Wind_Speed|General_Diffuse_Flows|Diffuse_Flows|Power_Zone_1|Power_Zone_2|      label|Month|Hour|\n",
      "+-----------+------------------+------------------+-----------+--------+----------+---------------------+-------------+------------+------------+-----------+-----+----+\n",
      "| 13875.8159|19757.010397789905|-5881.194497789906|      19.66|    75.7|     4.915|                6.853|        5.725| 20392.82392| 13401.26582| 13875.8159|    7|   6|\n",
      "|15245.34413|15191.932607670064|53.411522329935906|      19.35|    73.7|     0.071|                574.2|        571.0| 32356.72131| 20485.44892|15245.34413|    5|   9|\n",
      "|24932.98492|  24452.5108588495|480.47406115050035|      13.94|    70.1|     0.085|                 0.04|        0.126| 41711.18644| 24496.04863|24932.98492|    2|  21|\n",
      "|9483.282675| 8836.600516310064| 646.6821586899368|      21.13|   58.41|     0.085|                 10.4|         8.24| 25957.81182| 20475.93361|9483.282675|   10|   7|\n",
      "|11846.80851|10224.891169973198|1621.9173400268028|      19.05|    89.0|     0.268|                0.055|        0.133| 26071.24726| 15957.26141|11846.80851|   10|   3|\n",
      "+-----------+------------------+------------------+-----------+--------+----------+---------------------+-------------+------------+------------+-----------+-----+----+\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[Stage 11537:==>                                                (8 + 128) / 200]\r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Iteration 7/20: Wrote 5 rows to Power_Storage/stream_data_input/stream_data_part_06.csv\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "-------------------------------------------\n",
      "Batch: 5\n",
      "-------------------------------------------\n",
      "+-----------+------------------+-------------------+-----------+--------+----------+---------------------+-------------+------------+------------+-----------+-----+----+\n",
      "|      label|        prediction|           residual|Temperature|Humidity|Wind_Speed|General_Diffuse_Flows|Diffuse_Flows|Power_Zone_1|Power_Zone_2|      label|Month|Hour|\n",
      "+-----------+------------------+-------------------+-----------+--------+----------+---------------------+-------------+------------+------------+-----------+-----+----+\n",
      "|20256.40167| 23543.57596408233|-3287.1742940823315|       22.0|    59.9|     4.909|                0.289|         0.27| 25916.81063| 17232.91139|20256.40167|    7|   6|\n",
      "|16125.66802|14512.851725696502| 1612.8162943034968|      19.76|    69.6|     0.068|                480.9|        375.6| 30594.09836|  17000.6192|16125.66802|    5|  10|\n",
      "|17698.06452| 16562.18298455085|  1135.881535449149|      21.45|   46.03|     4.917|                632.7|        58.18| 32635.91489| 19635.36585|17698.06452|    3|  15|\n",
      "|26567.71084| 26338.30169431678| 229.40914568322114|      13.13|    75.4|     0.088|                0.062|        0.119| 43983.79747| 27326.44377|26567.71084|    1|  21|\n",
      "|22648.77743|21900.447604559933|  748.3298254400652|      25.81|    88.4|      4.91|                0.077|        0.093|  29400.9323| 20451.95354|22648.77743|    8|   2|\n",
      "+-----------+------------------+-------------------+-----------+--------+----------+---------------------+-------------+------------+------------+-----------+-----+----+\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[Stage 11540:>                                                  (2 + 128) / 200]\r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Iteration 8/20: Wrote 5 rows to Power_Storage/stream_data_input/stream_data_part_07.csv\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "-------------------------------------------\n",
      "Batch: 6\n",
      "-------------------------------------------\n",
      "+-----------+------------------+------------------+-----------+--------+----------+---------------------+-------------+------------+------------+-----------+-----+----+\n",
      "|      label|        prediction|          residual|Temperature|Humidity|Wind_Speed|General_Diffuse_Flows|Diffuse_Flows|Power_Zone_1|Power_Zone_2|      label|Month|Hour|\n",
      "+-----------+------------------+------------------+-----------+--------+----------+---------------------+-------------+------------+------------+-----------+-----+----+\n",
      "|12891.42857|13803.976393922323| -912.547823922323|       25.3|   47.29|     0.086|                 93.6|         76.5| 34591.50985| 21555.18672|12891.42857|   10|  17|\n",
      "| 11006.0024| 8775.303153181729|2230.6992468182707|       8.42|    81.8|     0.082|                0.081|        0.115| 23440.30418| 19375.26849| 11006.0024|   12|   1|\n",
      "|15730.12048|18095.456154427346|-2365.335674427346|      11.48|   64.11|     0.077|                31.36|        31.55| 31576.70886| 22030.39514|15730.12048|    1|  16|\n",
      "|19294.52308|21524.828862495535|-2230.305782495536|      25.06|    54.0|     0.071|                148.0|        122.1| 36772.45033| 21281.91268|19294.52308|    6|  17|\n",
      "|30481.00418| 28700.32347387056|1780.6807061294385|       22.8|   53.05|     4.907|                0.091|        0.119| 33290.63123| 23654.43038|30481.00418|    7|   1|\n",
      "+-----------+------------------+------------------+-----------+--------+----------+---------------------+-------------+------------+------------+-----------+-----+----+\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[Stage 11543:>                                                  (3 + 128) / 200]\r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Iteration 9/20: Wrote 5 rows to Power_Storage/stream_data_input/stream_data_part_08.csv\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "-------------------------------------------\n",
      "Batch: 7\n",
      "-------------------------------------------\n",
      "+-----------+------------------+-------------------+-----------+--------+----------+---------------------+-------------+------------+------------+-----------+-----+----+\n",
      "|      label|        prediction|           residual|Temperature|Humidity|Wind_Speed|General_Diffuse_Flows|Diffuse_Flows|Power_Zone_1|Power_Zone_2|      label|Month|Hour|\n",
      "+-----------+------------------+-------------------+-----------+--------+----------+---------------------+-------------+------------+------------+-----------+-----+----+\n",
      "|15919.59799|15595.637643526465| 323.96034647353554|       16.1|   41.69|     0.086|                278.9|        305.1| 30746.44068| 17686.32219|15919.59799|    2|  17|\n",
      "|22526.03077| 24408.77692535953|-1882.7461553595276|      21.53|    84.0|     0.065|                109.4|         84.9| 40523.44371| 25185.03119|22526.03077|    6|  18|\n",
      "|    16320.0|21215.520393098996| -4895.520393098996|      17.65|    71.3|     4.922|                10.96|        10.88| 40966.15385| 34553.30579|    16320.0|   11|  17|\n",
      "|27331.41066|24932.196121274254|  2399.214538725748|      28.66|   46.05|     0.072|                319.3|        318.6| 38875.20533| 26617.95143|27331.41066|    8|  17|\n",
      "|17054.45783| 18080.85208579989| -1026.394255799889|      14.58|   59.93|     0.082|                532.4|        591.8| 35981.77215| 22179.93921|17054.45783|    1|  14|\n",
      "+-----------+------------------+-------------------+-----------+--------+----------+---------------------+-------------+------------+------------+-----------+-----+----+\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[Stage 11546:>                                                  (1 + 128) / 200]\r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Iteration 10/20: Wrote 5 rows to Power_Storage/stream_data_input/stream_data_part_09.csv\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "-------------------------------------------\n",
      "Batch: 8\n",
      "-------------------------------------------\n",
      "+-----------+-----------------+-------------------+-----------+--------+----------+---------------------+-------------+------------+------------+-----------+-----+----+\n",
      "|      label|       prediction|           residual|Temperature|Humidity|Wind_Speed|General_Diffuse_Flows|Diffuse_Flows|Power_Zone_1|Power_Zone_2|      label|Month|Hour|\n",
      "+-----------+-----------------+-------------------+-----------+--------+----------+---------------------+-------------+------------+------------+-----------+-----+----+\n",
      "| 15960.1206|17679.43827452772|-1719.3176745277215|      11.17|   66.84|     0.083|                420.8|        395.5| 34425.76271| 21618.23708| 15960.1206|    2|  11|\n",
      "|9980.312125|10379.50383054603|-399.19170554602897|       15.9|   49.12|     0.083|                488.6|        111.5| 30150.57034| 25752.68487|9980.312125|   12|  12|\n",
      "|25351.22257|24570.68134210894|  780.5412278910626|      21.05|    74.6|     0.071|                0.073|        0.122| 33268.63485| 23508.34213|25351.22257|    8|   1|\n",
      "|22498.30721|22377.46821221859| 120.83899778140767|      26.06|    71.4|     4.906|                303.4|        170.1| 33927.10322| 26439.28194|22498.30721|    8|   9|\n",
      "|37373.72385|36393.40326043136|  980.3205895686406|       25.4|    69.0|     4.905|                0.102|         0.07| 47419.53488|  31955.6962|37373.72385|    7|  22|\n",
      "+-----------+-----------------+-------------------+-----------+--------+----------+---------------------+-------------+------------+------------+-----------+-----+----+\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[Stage 11549:>                                                  (0 + 128) / 200]\r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Iteration 11/20: Wrote 5 rows to Power_Storage/stream_data_input/stream_data_part_10.csv\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[Stage 11549:=>                                                 (7 + 128) / 200]\r"
     ]
    },
    {
     "ename": "KeyboardInterrupt",
     "evalue": "",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[16], line 33\u001b[0m\n\u001b[1;32m     30\u001b[0m     \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mIteration \u001b[39m\u001b[38;5;132;01m{\u001b[39;00mi\u001b[38;5;241m+\u001b[39m\u001b[38;5;241m1\u001b[39m\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m/20: Wrote \u001b[39m\u001b[38;5;132;01m{\u001b[39;00m\u001b[38;5;28mlen\u001b[39m(sampled_rows)\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m rows to \u001b[39m\u001b[38;5;132;01m{\u001b[39;00moutput_filepath\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m\"\u001b[39m)\n\u001b[1;32m     32\u001b[0m     \u001b[38;5;66;03m# Pause for 10 seconds\u001b[39;00m\n\u001b[0;32m---> 33\u001b[0m     \u001b[43mtime\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43msleep\u001b[49m\u001b[43m(\u001b[49m\u001b[38;5;241;43m10\u001b[39;49m\u001b[43m)\u001b[49m\n\u001b[1;32m     35\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mStreaming data simulation finished.\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[1;32m     36\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mYou can now observe the PySpark stream in your notebook console.\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n",
      "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
     ]
    }
   ],
   "source": [
    "# Define the path to the original streaming data file\n",
    "streaming_data_path = \"Power_Storage/power_streaming_data.csv\"\n",
    "# Define the input directory for the Spark Structured Stream\n",
    "streaming_input_dir = \"Power_Storage/stream_data_input\"\n",
    "\n",
    "# Ensure the input directory exists\n",
    "if not os.path.exists(streaming_input_dir):\n",
    "    os.makedirs(streaming_input_dir)\n",
    "    print(f\"Created streaming input directory: {streaming_input_dir}\")\n",
    "\n",
    "# Read the full streaming data into a pandas DataFrame\n",
    "try:\n",
    "    full_streaming_df = pd.read_csv(streaming_data_path)\n",
    "    print(f\"Successfully loaded {len(full_streaming_df)} rows from {streaming_data_path}\")\n",
    "except FileNotFoundError:\n",
    "    print(f\"Error: {streaming_data_path} not found. Please ensure the file is downloaded and placed in the correct location.\")\n",
    "    exit()\n",
    "\n",
    "print(\"Starting streaming data simulation...\")\n",
    "for i in range(20):\n",
    "    # Randomly sample five rows\n",
    "    sampled_rows = full_streaming_df.sample(n=5)\n",
    "\n",
    "    # Create a unique filename for the output CSV\n",
    "    output_filename = f\"stream_data_part_{i:02d}.csv\"\n",
    "    output_filepath = os.path.join(streaming_input_dir, output_filename)\n",
    "\n",
    "    # Write the sampled rows to a .csv file without indices, keeping headers\n",
    "    sampled_rows.to_csv(output_filepath, index=False, header=True)\n",
    "    print(f\"Iteration {i+1}/20: Wrote {len(sampled_rows)} rows to {output_filepath}\")\n",
    "\n",
    "    # Pause for 10 seconds\n",
    "    time.sleep(10)\n",
    "\n",
    "print(\"Streaming data simulation finished.\")\n",
    "print(\"You can now observe the PySpark stream in your notebook console.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ae54701b-23fc-49a4-a925-858e5474ee85",
   "metadata": {},
   "source": [
    "Create a short (one to three minute) video that shows you start the query, start the loop, and then\n",
    "watching the pyspark output update. This can easily be done with zoom. Please don’t make the video very\n",
    "long as I want you to upload it to Moodle! Note: If you don’t include the video you will lose substantial\n",
    "credit."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "869fc08d-37eb-481a-911d-07d033de8e93",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "-------------------------------------------\n",
      "Batch: 9\n",
      "-------------------------------------------\n",
      "+-----------+------------------+-------------------+-----------+--------+----------+---------------------+-------------+------------+------------+-----------+-----+----+\n",
      "|      label|        prediction|           residual|Temperature|Humidity|Wind_Speed|General_Diffuse_Flows|Diffuse_Flows|Power_Zone_1|Power_Zone_2|      label|Month|Hour|\n",
      "+-----------+------------------+-------------------+-----------+--------+----------+---------------------+-------------+------------+------------+-----------+-----+----+\n",
      "| 28540.9205|29886.108927266847|-1345.1884272668467|      26.31|    75.5|     4.921|                154.4|        150.5| 39210.09967| 24812.65823| 28540.9205|    7|  19|\n",
      "|28026.18182|26430.269988634038| 1595.9118313659637|      17.48|    75.9|     0.075|                0.311|        0.348| 43091.49623| 22729.12424|28026.18182|    4|  20|\n",
      "|17897.77324|18552.228233563488| -654.4549935634896|      22.82|   67.19|     4.924|                142.9|         85.7| 39555.39823| 23819.12682|17897.77324|    9|  18|\n",
      "|9035.294118|10522.343679253576| -1487.049561253576|      13.31|    75.1|     0.083|                309.1|        33.82| 29767.30038| 23510.27923|9035.294118|   12|  10|\n",
      "|14005.16129|13075.253988523415|  929.9073014765854|       8.54|    87.4|     0.073|                0.033|        0.156|  22525.2766|  13390.2439|14005.16129|    3|   4|\n",
      "+-----------+------------------+-------------------+-----------+--------+----------+---------------------+-------------+------------+------------+-----------+-----+----+\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "                                                                                \r"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "-------------------------------------------\n",
      "Batch: 10\n",
      "-------------------------------------------\n",
      "+-----------+------------------+------------------+-----------+--------+----------+---------------------+-------------+------------+------------+-----------+-----+----+\n",
      "|      label|        prediction|          residual|Temperature|Humidity|Wind_Speed|General_Diffuse_Flows|Diffuse_Flows|Power_Zone_1|Power_Zone_2|      label|Month|Hour|\n",
      "+-----------+------------------+------------------+-----------+--------+----------+---------------------+-------------+------------+------------+-----------+-----+----+\n",
      "|17881.44578|17469.495796742318| 411.9499832576803|      5.611|    73.7|     0.085|                0.062|          0.1| 28095.18987| 17992.70517|17881.44578|    1|   0|\n",
      "|18521.78138|16258.071914199016|2263.7094658009846|      25.04|   40.77|     0.076|                874.0|        52.61| 32923.27869|  19512.0743|18521.78138|    5|  12|\n",
      "|12943.82022|12447.094241678962| 496.7259783210375|      23.04|    83.7|     0.334|                336.8|        53.72|  30686.0177| 20076.92308|12943.82022|    9|   8|\n",
      "|22509.47368|23286.413166990413|-776.9394869904136|      19.49|    79.1|     0.077|                0.048|        0.141| 37820.85246|   21774.613|22509.47368|    5|   2|\n",
      "|13688.12308| 13432.23759065332| 255.8854893466796|      20.51|    74.8|     4.914|                0.077|        0.107| 20980.13245| 12360.49896|13688.12308|    6|   5|\n",
      "+-----------+------------------+------------------+-----------+--------+----------+---------------------+-------------+------------+------------+-----------+-----+----+\n",
      "\n"
     ]
    }
   ],
   "source": [
    "streamingQuery.stop()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "78472094-03d5-4b1a-98b6-9f531f9ef258",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "pySpark3",
   "language": "python",
   "name": "pyspark3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.15"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
