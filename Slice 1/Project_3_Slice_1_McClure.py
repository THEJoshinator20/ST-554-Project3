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
   "execution_count": 1,
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
   "execution_count": 2,
   "id": "d9f7a0de-bca4-4293-a504-c684f78cef6a",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING: Using incubator modules: jdk.incubator.vector\n",
      "Using Spark's default log4j profile: org/apache/spark/log4j2-defaults.properties\n",
      "Setting default log level to \"WARN\".\n",
      "To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).\n",
      "26/04/30 12:19:37 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable\n",
      "26/04/30 12:19:38 WARN Utils: Service 'SparkUI' could not bind on port 4040. Attempting port 4041.\n",
      "26/04/30 12:19:38 WARN Utils: Service 'SparkUI' could not bind on port 4041. Attempting port 4042.\n",
      "26/04/30 12:19:38 WARN Utils: Service 'SparkUI' could not bind on port 4042. Attempting port 4043.\n"
     ]
    },
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
      "First 5 rows of Spark DataFrame:\n"
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
   "execution_count": 3,
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
