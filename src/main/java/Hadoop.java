import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

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
	
	public static class KnnMapper extends Mapper<Object, Text, IntWritable, TwoDArrayWritable> {
		private int k; 
		private Instances testInstances;
		private int testSize;
		
		private TwoDArrayWritable outputMatrix = new TwoDArrayWritable(ArrayWritable.class);
		private ArrayWritable[][] classDistMatrix;
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException {
			super.setup(context);
		
			Configuration conf = context.getConfiguration();
			
			k = conf.getInt("k", 0);
			testSize = conf.getInt("testNumInstances", 0);
			
			BufferedReader reader = new BufferedReader(new FileReader(conf.get("testSetPath")));
			ArffReader arff = new ArffReader(reader);
			
			testInstances = arff.getData();
			
			classDistMatrix = new ArrayWritable[testSize][k];
			
			for(int i = 0; i < testSize; i++) {
				for(int j = 0; j < k; j++) {
					DoubleWritable[] classDistPair = new DoubleWritable[2];
					
					classDistPair[0].set(-1.);
					classDistPair[1].set(Double.MAX_VALUE);
					
					classDistMatrix[i][j].set(classDistPair);
				}
			}
			
			outputMatrix.set(classDistMatrix);
			System.out.println("MAP SETUP: " + context.getTaskAttemptID().getTaskID().getId());
		}

		@Override
		protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			if(value.toString().startsWith("@")) {
				return;
			}
			System.out.println("MAP^2");
			
			String[] valueTokens = value.toString().split(",");
			int trainAttrSize = valueTokens.length;
			
			double[] trainInstance = new double[trainAttrSize];
			
			for(int i = 0; i < trainAttrSize; i++) {
				trainInstance[i] = Double.parseDouble(valueTokens[i]);
			}
			
			DoubleWritable[] tempPair = new DoubleWritable[2];
			boolean newMinDistFound;
			
			for(int i = 0; i < testSize; i++) {
				double[] testInstance = testInstances.get(i).toDoubleArray(); 
				
				double dist = calculateDistance(trainInstance, testInstance);
				
				newMinDistFound = false;
				
				for(int j = 0; j < k; j++) {
					DoubleWritable[] classDistPair = (DoubleWritable[]) classDistMatrix[i][j].get();
					
					if(!newMinDistFound) {
						double currentNeighborDist = classDistPair[1].get();
						
						if(dist < currentNeighborDist) {
							newMinDistFound = true;
							tempPair = classDistPair;
							
							classDistPair[0].set(trainInstance[trainAttrSize - 1]);
							classDistPair[1].set(dist);
							
							classDistMatrix[i][j].set(classDistPair);
						}
					} else {
						DoubleWritable[] anotherTemp = classDistPair;
						
						classDistPair[0].set(tempPair[0].get());
						classDistPair[1].set(tempPair[1].get());
						
						tempPair = anotherTemp;
					}
				}
			}
		}

		@Override
		protected void cleanup(Context context) throws IOException, InterruptedException {
			IntWritable outputKey = new IntWritable(context.getTaskAttemptID().getTaskID().getId());
			outputMatrix.set(classDistMatrix);
			
			context.write(outputKey, outputMatrix);
			
			System.out.println("MAP CLEAN: " + outputKey);
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
	
	public static class KnnReducer extends Reducer<IntWritable, TwoDArrayWritable, IntWritable, IntWritable> {
		private int k;
		private int testSize;
		private int numClasses;
		
		private ArrayWritable[][] finalClassDistMatrix;
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException {
			super.setup(context);
			
			Configuration conf = context.getConfiguration();
			
			k = conf.getInt("k", 0);
			testSize = conf.getInt("testNumInstances", 0);
			numClasses = conf.getInt("numClasses", 0);
			
			finalClassDistMatrix = new ArrayWritable[testSize][k];
			
			for(int i = 0; i < testSize; i++) {
				for(int j = 0; j < k; j++) {
					DoubleWritable[] classDistPair = new DoubleWritable[2];
					
					classDistPair[0].set(-1.);
					classDistPair[1].set(Double.MAX_VALUE);
					
					finalClassDistMatrix[i][j].set(classDistPair);
				}
			}
			
			System.out.println("REDUCE SETUP: " + context.getTaskAttemptID().getTaskID().getId());
		}
		
		@Override
		protected void reduce(IntWritable key, Iterable<TwoDArrayWritable> value, Context context) throws IOException, InterruptedException {
			System.out.println("REDUCE^2");
			
			int testIndex = 0;
			DoubleWritable[] tempPair = new DoubleWritable[2];
			boolean newMinDistFound;
			
			for(TwoDArrayWritable val : value) {
				Writable[][] row = val.get();
				
				newMinDistFound = false;
				
				for(int j = 0; j < k; j++) {
					DoubleWritable[] currentClassDistPair = (DoubleWritable[]) row[j];
					DoubleWritable[] finalClassDistPair = (DoubleWritable[]) finalClassDistMatrix[testIndex][j].get();
					
					if(!newMinDistFound) {
						double currentDist = currentClassDistPair[1].get();
						double finalDist = finalClassDistPair[1].get();
						
						if(currentDist < finalDist) {
							newMinDistFound = true;
							tempPair = finalClassDistPair;
							
							finalClassDistPair[0].set(finalClassDistPair[0].get());
							finalClassDistPair[1].set(currentDist);
							
							finalClassDistMatrix[testIndex][j].set(finalClassDistPair);
						}
					} else {
						DoubleWritable[] anotherTemp = finalClassDistPair;
						
						finalClassDistPair[0].set(tempPair[0].get());
						finalClassDistPair[1].set(tempPair[1].get());
						
						tempPair = anotherTemp;
					}
				}
				
				testIndex++;
			}
		}

		@Override
		protected void cleanup(Context context) throws IOException, InterruptedException {
			for(int i = 0; i < testSize; i++) {
				int[] classCounts = new int[numClasses];
				int predictedClass = -1;
				int highestClassCount = 0;
				
				for(int j = 0; j < k; j++) {
					DoubleWritable[] classDistPair = (DoubleWritable[]) finalClassDistMatrix[i][j].get();
					int currentClass = (int) classDistPair[0].get();
					
					classCounts[currentClass]++;
					
					if(classCounts[currentClass] > highestClassCount) {
						predictedClass = currentClass;
						highestClassCount = classCounts[currentClass];
					}
				}
				
				context.write(new IntWritable(i), new IntWritable(predictedClass));
			}
			
			System.out.println("REDUCE CLEAN: " + context.getTaskAttemptID().getTaskID().getId());
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
		
		BufferedReader reader = new BufferedReader(new FileReader(args[1]));
		ArffReader arff = new ArffReader(reader, 20);
		Instances data = arff.getStructure();
		
		conf.setInt("testNumInstances", data.numInstances());
		
		reader = new BufferedReader(new FileReader(args[0]));
		arff = new ArffReader(reader, 20);
		data = arff.getStructure();
		data.setClassIndex(data.numAttributes() - 1);
		
		conf.setInt("numClasses", data.numClasses());
		
		Job job = Job.getInstance(conf, "knn");
		job.setJarByClass(Hadoop.class);
		job.setMapperClass(KnnMapper.class);
		job.setReducerClass(KnnReducer.class);
		job.setNumReduceTasks(1);
		
		job.setMapOutputKeyClass(IntWritable.class);
		job.setMapOutputValueClass(TwoDArrayWritable.class);
		job.setOutputKeyClass(IntWritable.class);
		job.setOutputValueClass(IntWritable.class);
		
		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[3]));
		
		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}
	
}
