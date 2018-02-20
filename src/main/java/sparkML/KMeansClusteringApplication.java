package sparkML;

import org.apache.commons.lang.time.StopWatch;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.DataFrameReader;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.SQLContext;

public class KMeansClusteringApplication {

	private static SparkConf config;
	private static JavaSparkContext javaSparkContext;
	private static SQLContext sqlContext;
	private static Dataset data;

	private static void assembleFeaturesInFeaturesColumn(final String[] columns) {
		VectorAssembler assembler = new VectorAssembler().setInputCols(columns).setOutputCol("irisFeatures");
		data = assembler.transform(data);
	}

	private static KMeansModel getKMeansModel(final Dataset data) {
		KMeans kMeans = new KMeans().setK(3).setFeaturesCol("irisFeatures").setMaxIter(100);
		return kMeans.fit(data);
	}

	private static SQLContext initConfiguration() {
		config = new SparkConf().setAppName("KMeansClusteringApplication").setMaster("local");
		javaSparkContext = new JavaSparkContext(config);
		javaSparkContext.setLogLevel("ERROR");
		return new SQLContext(javaSparkContext);
	}

	public static void main(String[] args) {

		sqlContext = initConfiguration();
		StopWatch stopWatch = new StopWatch();

		DataFrameReader csvReader = sqlContext.read().format("csv").option("header", "true").option("inferSchema",
				"true");

		data = csvReader.load("data/Iris.csv");

		assembleFeaturesInFeaturesColumn(
				new String[] { "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm" });

		stopWatch.start();
		KMeansModel kMeansModel = getKMeansModel(data);
		stopWatch.stop();

		System.out.println("Time for creating/training the K-Means model: " + stopWatch.getTime() + " milliseconds.");
	}
}
