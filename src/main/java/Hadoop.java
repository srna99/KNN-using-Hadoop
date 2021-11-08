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
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class Hadoop {
	
	public static class KNNMapper extends Mapper<Object, Text, IntWritable, TwoDArrayWritable> {
		private int k; 
		private Instances testInstances;
		private int testSize;
		
		private IntWritable outputKey;
		private static TwoDArrayWritable outputMatrix = new TwoDArrayWritable(ArrayWritable.class);
		
		private static ArrayWritable[][] distanceMatrix;
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException {
			super.setup(context);
			
			Configuration conf = context.getConfiguration();
			
			k = conf.getInt("k", 0);
			
			BufferedReader reader = new BufferedReader(new FileReader(conf.get("testSetPath")));
			ArffReader arff = new ArffReader(reader);
			
			testInstances = arff.getData();
			testSize = arff.getStructure().numInstances();
			
			distanceMatrix = new ArrayWritable[testSize][k];
			
			for(int i = 0; i < testSize; i++) {
				for(int j = 0; j < k; j++) {
					DoubleWritable[] classDistPair = new DoubleWritable[2];
					
					classDistPair[0].set(-1.);
					classDistPair[1].set(Double.MAX_VALUE);
					
					distanceMatrix[i][j].set(classDistPair);
				}
			}
			
			outputMatrix.set(distanceMatrix);
		}

		@Override
		protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			if(value.toString().startsWith("@")) {
				return;
			}
			System.out.println("TEST");
			String[] valueTokens = value.toString().split(",");
			int trainAttrSize = valueTokens.length;
			
			double[] trainInstance = new double[trainAttrSize];
			
			for(int i = 0; i < trainAttrSize; i++) {
				trainInstance[i] = Double.parseDouble(valueTokens[i]);
			}
			
			double[] testInstance;
			double currentNeighborDist;
			double dist;
			DoubleWritable[] classDistPair;
			
			for(int i = 0; i < testSize; i++) {
				testInstance = testInstances.get(i).toDoubleArray(); 
				
				dist = calculateDistance(trainInstance, testInstance);
				
				for(int j = 0; j < k; j++) {
					classDistPair = (DoubleWritable[]) distanceMatrix[i][j].get();
					
					currentNeighborDist = Double.parseDouble(classDistPair[1].toString());
					
					if(dist < currentNeighborDist) {
						classDistPair[0].set(trainInstance[trainAttrSize - 1]);
						classDistPair[1].set(dist);
						
						distanceMatrix[i][j].set(classDistPair);
						
						break;
					}
				}
			}
		}

		@Override
		protected void cleanup(Context context) throws IOException, InterruptedException {
			outputKey.set(context.getTaskAttemptID().getTaskID().getId());
			outputMatrix.set(distanceMatrix);
			
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
	
	public static class KNNReducer extends Reducer<IntWritable, TwoDArrayWritable, IntWritable, IntWritable> {
		@Override
		protected void setup(Context context) throws IOException, InterruptedException {
			super.setup(context);
		}
		
		@Override
		protected void reduce(IntWritable key, Iterable<TwoDArrayWritable> value, Context context) throws IOException, InterruptedException {
			
		}

		@Override
		protected void cleanup(Context context) throws IOException, InterruptedException {
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
		
		Job job = Job.getInstance(conf, "knn");
		job.setJarByClass(Hadoop.class);
		job.setMapperClass(KNNMapper.class);
		job.setReducerClass(KNNReducer.class);
		job.setNumReduceTasks(1);
		job.setOutputKeyClass(IntWritable.class);
		job.setOutputValueClass(IntWritable.class);
		
		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[3]));
		
		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}
	
}
