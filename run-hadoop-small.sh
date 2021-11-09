rm -R predictions
hadoop jar target/KNN-MapReduce-0.0.1-SNAPSHOT-jar-with-dependencies.jar Hadoop datasets/small-train.arff datasets/small-test.arff 3 predictions