import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.TwoDArrayWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class Hadoop {
	
	public static class PairTwoDArrayWritable extends TwoDArrayWritable {
	    public PairTwoDArrayWritable() {
	        super(DoubleArrayWritable.class);
	    }
	    
	    public PairTwoDArrayWritable(DoubleArrayWritable[][] values) {
	        super(DoubleArrayWritable.class, values);
	    }
	}
	
	public static class DoubleArrayWritable extends ArrayWritable{
		public DoubleArrayWritable() {
			super(DoubleWritable.class);
		}
		
		public DoubleArrayWritable(DoubleWritable[] values) {
			super(DoubleWritable.class, values);
		}
	}
	
	public static class KnnMapper extends Mapper<Object, Text, IntWritable, PairTwoDArrayWritable> {
		private int k; 
		private Instances testInstances;
		private int testSize;
		
		private DoubleArrayWritable[][] classDistMatrix;
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException {
			super.setup(context);
		
			Configuration conf = context.getConfiguration();
			
			k = conf.getInt("k", 0);
			testSize = conf.getInt("testNumInstances", 0);
			
			String path = new File("").getAbsolutePath();
			BufferedReader reader = new BufferedReader(new FileReader(path + System.getProperty("file.separator") + conf.get("testSetPath")));
			ArffReader arff = new ArffReader(reader);
			
			testInstances = arff.getData();
			reader.close();
			
			classDistMatrix = new DoubleArrayWritable[testSize][k];
			
			for(int i = 0; i < testSize; i++) {
				for(int j = 0; j < k; j++) {
					DoubleWritable[] classDistPair = new DoubleWritable[2];
					
					classDistPair[0] = new DoubleWritable(-1);
					classDistPair[1] = new DoubleWritable(Double.MAX_VALUE);
					
					classDistMatrix[i][j] = new DoubleArrayWritable(classDistPair);
				}
			}
		}

		@Override
		protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			if(value.toString().startsWith("@")) {
				return;
			}
			
			String[] valueTokens = value.toString().split(",");
			int trainAttrSize = valueTokens.length;
			
			double[] trainInstance = new double[trainAttrSize];
			
			for (int i = 0; i < trainAttrSize; i++) {
				trainInstance[i] = Double.parseDouble(valueTokens[i]);
			}
			
			boolean newMinDistFound;
			
			for (int i = 0; i < testSize; i++) {
				double[] testInstance = testInstances.get(i).toDoubleArray();
				
				double dist = calculateDistance(trainInstance, testInstance);
				
				DoubleWritable[] tempPair = new DoubleWritable[2];
				newMinDistFound = false;

				for (int j = 0; j < k; j++) {
					DoubleWritable[] classDistPair = (DoubleWritable[]) classDistMatrix[i][j].get();

					if (!newMinDistFound) {
						double currentNeighborDist = classDistPair[1].get();
						
						if (dist < currentNeighborDist) {
							newMinDistFound = true;
							tempPair[0] = new DoubleWritable(classDistPair[0].get());
							tempPair[1] = new DoubleWritable(classDistPair[1].get());

							classDistPair[0].set(trainInstance[trainAttrSize - 1]);
							classDistPair[1].set(dist);

							classDistMatrix[i][j].set(classDistPair);
						}
					} else {
						DoubleWritable[] anotherTemp = new DoubleWritable[2];
						anotherTemp[0] = new DoubleWritable(classDistPair[0].get());
						anotherTemp[1] = new DoubleWritable(classDistPair[1].get());
						
						classDistPair[0].set(tempPair[0].get());
						classDistPair[1].set(tempPair[1].get());
						
						classDistMatrix[i][j].set(classDistPair);
						
						tempPair[0].set(anotherTemp[0].get());
						tempPair[1].set(anotherTemp[1].get());
					}
				}
			}
		}

		@Override
		protected void cleanup(Context context) throws IOException, InterruptedException {
			IntWritable outputKey = new IntWritable(context.getTaskAttemptID().getTaskID().getId());
			PairTwoDArrayWritable outputMatrix = new PairTwoDArrayWritable(classDistMatrix);
			
			context.write(outputKey, outputMatrix);
			
			super.cleanup(context);
		}
		
		private double calculateDistance(double[] a, double[] b) {
			double sum = 0;
			double diff;
		    
		    for (int i = 0; i < a.length - 1; i++) {
		        diff = a[i] - b[i];
		        sum += diff * diff;
		    }
		    
		    return sum;
		}
	}
	
	public static class KnnReducer extends Reducer<IntWritable, PairTwoDArrayWritable, IntWritable, IntWritable> {
		private int k;
		private int testSize;
		
		private double[][][] finalClassDistMatrix;
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException {
			super.setup(context);
			
			Configuration conf = context.getConfiguration();
			
			k = conf.getInt("k", 0);
			testSize = conf.getInt("testNumInstances", 0);
			
			finalClassDistMatrix = new double[testSize][k][2];
			
			for (int i = 0; i < testSize; i++) {
				for (int j = 0; j < k; j++) {
					finalClassDistMatrix[i][j][0] = -1;
					finalClassDistMatrix[i][j][1] = Double.MAX_VALUE;
				}
			}
		}
		
		@Override
		protected void reduce(IntWritable key, Iterable<PairTwoDArrayWritable> value, Context context) throws IOException, InterruptedException {
			for (PairTwoDArrayWritable val : value) {
				DoubleArrayWritable[][] matrix = new DoubleArrayWritable[testSize][k];
				matrix = (DoubleArrayWritable[][]) val.toArray();
				
				for(int i = 0; i < testSize; i++) {
					double[] tempPair = new double[2];
					
					for (int j = 0; j < k; j++) {
						Writable[] currentClassDistPair = new Writable[2];
						currentClassDistPair = matrix[i][j].get();
						
						double[] finalClassDistPair = finalClassDistMatrix[i][j];
						
						double currentDist = Double.parseDouble(currentClassDistPair[1].toString());
						double finalDist = finalClassDistPair[1];
						
						if (currentDist < finalDist) {
							tempPair[0] = finalClassDistPair[0];
							tempPair[1] = finalClassDistPair[1];

							finalClassDistPair[0]= Double.parseDouble(currentClassDistPair[0].toString());
							finalClassDistPair[1] = currentDist;
							
							finalClassDistMatrix[i][j] = finalClassDistPair;
							
							for(int m = j + 1; m < k; m++) {
								double[] shiftPair = finalClassDistMatrix[i][m];
								
								double[] anotherTemp = new double[2];
								anotherTemp[0] = shiftPair[0];
								anotherTemp[1] = shiftPair[1];
								
								shiftPair[0] = tempPair[0];
								shiftPair[1] = tempPair[1];
								
								finalClassDistMatrix[i][m] = shiftPair;
								
								tempPair[0] = anotherTemp[0];
								tempPair[1] = anotherTemp[1];
							}
						}
					}
				}
			} 
		}

		@Override
		protected void cleanup(Context context) throws IOException, InterruptedException {
			for (int i = 0; i < testSize; i++) {
				Map<Integer, Integer> classCounts = new HashMap<Integer, Integer>();
				
				for (int j = 0; j < k; j++) {
					double[] classDistPair = finalClassDistMatrix[i][j];
					int currentClass = (int) classDistPair[0];
					
					if(!classCounts.containsKey(currentClass)) {
						classCounts.put(currentClass, 0);
					} else {
						classCounts.put(currentClass, classCounts.get(currentClass) + 1);
					}
				}
				
				int predictedClass = -1;
				int highestClassCount = 0;
				
				for (Map.Entry<Integer, Integer> entry : classCounts.entrySet()) {
					int key = entry.getKey();
					int val = entry.getValue();
					
					if(val > highestClassCount) {
						predictedClass = key;
						highestClassCount = val;
					}
				}
				
				context.write(new IntWritable(i), new IntWritable(predictedClass));
			}
			
			super.cleanup(context);
		}
	}

	public static void main(String[] args) throws Exception {
		if (args.length != 4) {
            System.err.println("Usage: Hadoop datasets/train.arff datasets/test.arff k output");
            System.exit(2);
        }
		
		Configuration conf = new Configuration();
		
		conf.set("testSetPath", args[1]);
		conf.setInt("k", Integer.parseInt(args[2]));
		
		String path = new File("").getAbsolutePath();
		BufferedReader reader = new BufferedReader(new FileReader(path + System.getProperty("file.separator") + args[1]));
		ArffReader arff = new ArffReader(reader);
		Instances data = arff.getData();
		
		int numInstances = data.numInstances();
		conf.setInt("testNumInstances", numInstances);
		
		Job job = Job.getInstance(conf, "knn");
		job.setJarByClass(Hadoop.class);
		job.setMapperClass(KnnMapper.class);
		job.setReducerClass(KnnReducer.class);
		job.setNumReduceTasks(1);
		
		job.setMapOutputKeyClass(IntWritable.class);
		job.setMapOutputValueClass(PairTwoDArrayWritable.class);
		job.setOutputKeyClass(IntWritable.class);
		job.setOutputValueClass(IntWritable.class);
		
		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[3]));
		
		long startTime = System.nanoTime();
		
		job.waitForCompletion(true);
		
		long endTime = System.nanoTime();

		long milliseconds = (endTime - startTime) / 1000000;
		
		try {
			File predictions = new File(path + System.getProperty("file.separator") + "predictions/part-r-00000");
			Scanner sc = new Scanner(predictions);
			
			reader = new BufferedReader(new FileReader(path + System.getProperty("file.separator") + args[1]));
			arff = new ArffReader(reader, 1000);
			data = arff.getData();
			data.setClassIndex(data.numAttributes() - 1);
			
			Instance inst;
			int correct = 0;
			
			while(sc.hasNextLine() && (inst = arff.readInstance(data)) != null) {
				sc.nextInt();
				int prediction = sc.nextInt();
				
				int trueValue = (int) inst.classValue();
				
				if(prediction == trueValue) {
					correct++;
				}
			}
			sc.close();
			reader.close();
			
			double accuracy = (correct * 1.0) / numInstances;
			System.out.println("It took " + milliseconds + " ms and results have an accuracy of " + accuracy);
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	
}
