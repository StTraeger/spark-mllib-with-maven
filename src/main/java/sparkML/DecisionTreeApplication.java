package sparkML;

import java.util.HashMap;
import java.util.Map;

import org.apache.commons.lang.time.StopWatch;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.DataFrameReader;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.types.StructType;

public class DecisionTreeApplication {

	private static SparkConf config;
	private static JavaSparkContext javaSparkContext;
	private static SQLContext sqlContext;
	private static Dataset data;

	private static void assembleFeaturesInFeaturesColumn(final String[] featureColumns) {
		VectorAssembler assembler = new VectorAssembler().setInputCols(featureColumns).setOutputCol("features");
		data = assembler.transform(data);
	}

	private static void changeColumnTypeStringToInt(final String[] columns) {
		for (String columnName : columns) {
			final String tmpColumn = columnName + "Tmp";
			data = data.withColumn(tmpColumn, data.col(columnName).cast("int")).drop(columnName)
					.withColumnRenamed(tmpColumn, columnName);
		}
	}

	private static Dataset encodeIndexedColumn(final String columnName, final Dataset dataFrame) {
		String colNameUpperCase = columnName.substring(0, 1).toUpperCase() + columnName.substring(1).toLowerCase();
		OneHotEncoder encoder = new OneHotEncoder().setInputCol("indexed" + colNameUpperCase)
				.setOutputCol("vectorized" + colNameUpperCase);
		return encoder.transform(dataFrame);
	}

	private static DecisionTreeClassificationModel getModel(Dataset trainingSet) {
		DecisionTreeClassifier decisionTree = new DecisionTreeClassifier().setFeaturesCol("features")
				.setLabelCol("indexedIncome").setMaxBins(100);
		return decisionTree.fit(trainingSet);
	}

	private static void handleMissingValues() {
		Map<String, String> replaceMap = new HashMap<>();
		replaceMap.put("?", "");

		data = data.drop("workclass", "education", "occupation", "relationship", "race", "native.country");
		data = data.na().replace(data.columns(), replaceMap);
		data = data.na().drop();
	}

	private static void indexAndEncodeColumns(final String[] columns) {
		for (String column : columns) {
			data = indexCategoricalColumn(column, data);
			data = encodeIndexedColumn(column, data);
		}
	}

	private static Dataset indexCategoricalColumn(final String columnName, final Dataset dataFrame) {
		String colNameUpperCase = columnName.substring(0, 1).toUpperCase() + columnName.substring(1).toLowerCase();
		StringIndexerModel categoricalIndexer = new StringIndexer().setInputCol(columnName)
				.setOutputCol("indexed" + colNameUpperCase).fit(dataFrame);
		return categoricalIndexer.transform(dataFrame);
	}

	private static SQLContext initConfiguration() {
		config = new SparkConf().setAppName("DecisionTreeApplication").setMaster("local");
		javaSparkContext = new JavaSparkContext(config);
		javaSparkContext.setLogLevel("ERROR");
		return new SQLContext(javaSparkContext);
	}

	public static void main(String[] args) {

		sqlContext = initConfiguration();

		StopWatch stopWatch = new StopWatch();

		DataFrameReader csvReader = sqlContext.read().format("csv").option("header", "true").option("inferSchema",
				"true");
		data = csvReader.load("data/adult.csv");

		handleMissingValues();

		StructType schema = data.schema();
		System.out.println(schema.treeString());

		indexAndEncodeColumns(new String[] { "income", "maritalstatus", "sex" });
		changeColumnTypeStringToInt(new String[] { "hoursperweek", "age" });
		assembleFeaturesInFeaturesColumn(
				new String[] { "age", "educationnum", "vectorizedMaritalstatus", "vectorizedSex", "hoursperweek" });

		Dataset[] splits = data.randomSplit(new double[] { 0.7, 0.3 });
		Dataset training = splits[0];
		Dataset testing = splits[1];

		stopWatch.start();
		DecisionTreeClassificationModel model = getModel(training);
		stopWatch.stop();

		System.out.println("Time for creating/training the model: " + stopWatch.getTime() + " Milliseconds.");
	}
}
