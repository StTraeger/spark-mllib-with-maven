package sparkML;

import org.apache.commons.lang.time.StopWatch;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.DataFrameReader;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.SQLContext;

public class LinearRegressionApplication {

	private static SparkConf config;
	private static JavaSparkContext javaSparkContext;
	private static SQLContext sqlContext;
	private static Dataset bitcoinData;

	private static void assembleFeaturesInFeaturesColumn(final String[] columns) {
		VectorAssembler assembler = new VectorAssembler().setInputCols(columns).setOutputCol("features");
		bitcoinData = assembler.transform(bitcoinData);
	}

	private static LinearRegressionModel getLinearRegressionModel(Dataset training) {
		LinearRegression linearRegression = new LinearRegression().setLabelCol("Weighted_Price")
				.setFeaturesCol("features");
		return linearRegression.fit(training);
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

		bitcoinData = csvReader.load("data/bitstampUSD_1-min_data_2012-01-01_to_2017-10-20.csv");

		bitcoinData = bitcoinData.na().drop();
		bitcoinData.schema().printTreeString();

		assembleFeaturesInFeaturesColumn(
				new String[] { "Open", "High", "Low", "Close", "Volume_(BTC)", "Volume_(Currency)" });

		Dataset[] splits = bitcoinData.randomSplit(new double[] { 0.7, 0.3 });
		Dataset training = splits[0];
		Dataset testing = splits[1];

		stopWatch.start();
		LinearRegressionModel regressionModel = getLinearRegressionModel(training);
		stopWatch.stop();

		System.out.println("Time for creating/training the model: " + stopWatch.getTime() + " milliseconds.");
	}

}
