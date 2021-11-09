rm -R predictions
hadoop jar target/KNN-MapReduce-0.0.1-SNAPSHOT-jar-with-dependencies.jar Hadoop datasets/medium-train.arff datasets/medium-test.arff 3 predictions