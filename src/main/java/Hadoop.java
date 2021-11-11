import weka.core.AttributeStats;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffLoader.ArffReader;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

import org.apache.commons.lang3.ArrayUtils;
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
		
//		private TwoDArrayWritable outputMatrix = new TwoDArrayWritable(ArrayWritable.class);
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
			
			System.out.println(k + ", " + testSize);
			
			classDistMatrix = new DoubleArrayWritable[testSize][k];
			
			for(int i = 0; i < testSize; i++) {
				for(int j = 0; j < k; j++) {
					DoubleWritable[] classDistPair = new DoubleWritable[2];
					
					classDistPair[0].set(-1.); //ERR
					classDistPair[1].set(Double.MAX_VALUE); 
					System.out.println(classDistPair);
					
					classDistMatrix[i][j].set(classDistPair);
				}
			}
			
//			outputMatrix.set(classDistMatrix);
			System.out.println("MAP SETUP: " + context.getTaskAttemptID().getTaskID().getId());
			System.out.println(classDistMatrix);
			System.out.println("1. " + classDistMatrix[0][0].get()[1].toString());
		}

		@Override
		protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			if(value.toString().startsWith("@")) {
				return;
			}
			System.out.println("MAP^2");
			
			try {
				String[] valueTokens = value.toString().split(",");
				int trainAttrSize = valueTokens.length;
				System.out.println("1. " + trainAttrSize);
				double[] trainInstance = new double[trainAttrSize];
				for (int i = 0; i < trainAttrSize; i++) {
					trainInstance[i] = Double.parseDouble(valueTokens[i]);
				}
				DoubleWritable[] tempPair = new DoubleWritable[2];
				boolean newMinDistFound;
				for (int i = 0; i < testSize; i++) {
					double[] testInstance = testInstances.get(i).toDoubleArray();
					System.out.println("2. " + testInstance.length);
					double dist = calculateDistance(trainInstance, testInstance);
					System.out.println("3. " + dist);
					newMinDistFound = false;

					for (int j = 0; j < k; j++) {
						DoubleWritable[] classDistPair = (DoubleWritable[]) classDistMatrix[i][j].get();

						if (!newMinDistFound) {
							double currentNeighborDist = classDistPair[1].get();
							System.out.println("4. " + currentNeighborDist);
							if (dist < currentNeighborDist) {
								newMinDistFound = true;
								tempPair = classDistPair;

								classDistPair[0].set(trainInstance[trainAttrSize - 1]);
								classDistPair[1].set(dist);

								classDistMatrix[i][j].set(classDistPair);
								System.out.println("5. " + classDistMatrix[i][j].get()[1].toString());
							}
						} else {
							DoubleWritable[] anotherTemp = classDistPair;

							classDistPair[0].set(tempPair[0].get());
							classDistPair[1].set(tempPair[1].get());
							
							classDistMatrix[i][j].set(classDistPair);

							tempPair = anotherTemp;
						}
					}
				} 
				System.out.println("6. " + classDistMatrix.length);
			} catch (Exception e) {
				System.err.println("MAP^2 ERR: " + e.getStackTrace());
			}
		}

		@Override
		protected void cleanup(Context context) throws IOException, InterruptedException {
			try {
				IntWritable outputKey = new IntWritable(context.getTaskAttemptID().getTaskID().getId());
//				outputMatrix.set(classDistMatrix);
				PairTwoDArrayWritable outputMatrix = new PairTwoDArrayWritable(classDistMatrix);
				context.write(outputKey, outputMatrix);
				DoubleArrayWritable d = (DoubleArrayWritable) outputMatrix.get()[0][0];
				System.out.println("MAP CLEAN: " + outputKey);
				System.out.println("1. " + d.get()[1].toString() + ", " + classDistMatrix[0][0].get()[1].toString());
			} catch (Exception e) {
				System.err.println("MAP CLEAN ERR: " + e.getStackTrace());
			}
			
			
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
//		private int numClasses;
		
		private ArrayWritable[][] finalClassDistMatrix;
		private Set<Double> distinctClasses = new HashSet<Double>();
		
		@Override
		protected void setup(Context context) throws IOException, InterruptedException {
			try {
				super.setup(context);
				Configuration conf = context.getConfiguration();
				k = conf.getInt("k", 0);
				testSize = conf.getInt("testNumInstances", 0);
//				numClasses = conf.getInt("numClasses", 0);
				finalClassDistMatrix = new ArrayWritable[testSize][k];
				for (int i = 0; i < testSize; i++) {
					for (int j = 0; j < k; j++) {
						DoubleWritable[] classDistPair = new DoubleWritable[2];

						classDistPair[0].set(-1.);
						classDistPair[1].set(Double.MAX_VALUE);

						finalClassDistMatrix[i][j].set(classDistPair);
					}
				}
				System.out.println("REDUCE SETUP: " + context.getTaskAttemptID().getTaskID().getId());
			} catch (Exception e) {
				System.err.println("REDUCE SETUP ERR: " + e.getStackTrace());
			}
		}
		
		@Override
		protected void reduce(IntWritable key, Iterable<PairTwoDArrayWritable> value, Context context) throws IOException, InterruptedException {
			System.out.println("REDUCE^2");
			
			try {
				int testIndex = 0;
				DoubleWritable[] tempPair = new DoubleWritable[2];
				boolean newMinDistFound;
				
				for (PairTwoDArrayWritable val : value) {
					System.out.println("1. " + testIndex);
					ArrayWritable[][] row = (ArrayWritable[][]) val.get();
					
					newMinDistFound = false;

					for(int i = 0; i < testSize; i++) {
						for (int j = 0; j < k; j++) {
							DoubleWritable[] currentClassDistPair = (DoubleWritable[]) row[i][j].get();
							DoubleWritable[] finalClassDistPair = (DoubleWritable[]) finalClassDistMatrix[i][j].get();
							System.out.println("2. " + currentClassDistPair.length);
							if (!newMinDistFound) {
								double currentDist = currentClassDistPair[1].get();
								double finalDist = finalClassDistPair[1].get();
	
								if (currentDist < finalDist) {
									newMinDistFound = true;
									tempPair = finalClassDistPair;
	
									finalClassDistPair[0].set(currentClassDistPair[0].get());
									finalClassDistPair[1].set(currentDist);
									System.out.println("3. " + tempPair[1]);
									finalClassDistMatrix[i][j].set(finalClassDistPair);
									
									distinctClasses.add(currentClassDistPair[0].get());
								}
							} else {
								DoubleWritable[] anotherTemp = finalClassDistPair;
	
								finalClassDistPair[0].set(tempPair[0].get());
								finalClassDistPair[1].set(tempPair[1].get());
								
								finalClassDistMatrix[i][j].set(finalClassDistPair);
								
								System.out.println("4. " + anotherTemp[1]);
								tempPair = anotherTemp;
							}
						}
					}

					testIndex++;
				} 
			} catch (Exception e) {
				System.err.println("REDUCE^2 ERR: " + e.getStackTrace());
			}
		}

		@Override
		protected void cleanup(Context context) throws IOException, InterruptedException {
			try {
				for (int i = 0; i < testSize; i++) {
					int[] classCounts = new int[distinctClasses.size()];
					System.out.println("NumClasses: " + distinctClasses.size());
					int predictedClass = -1;
					int highestClassCount = 0;

					for (int j = 0; j < k; j++) {
						DoubleWritable[] classDistPair = (DoubleWritable[]) finalClassDistMatrix[i][j].get();
						int currentClass = (int) classDistPair[0].get();

						classCounts[currentClass]++;

						if (classCounts[currentClass] > highestClassCount) {
							predictedClass = currentClass;
							highestClassCount = classCounts[currentClass];
						}
					}

					context.write(new IntWritable(i), new IntWritable(predictedClass));
				}
				System.out.println("REDUCE CLEAN: " + context.getTaskAttemptID().getTaskID().getId());
			} catch (Exception e) {
				System.err.println("REDUCE CLEAN ERR: " + e.getStackTrace());
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
//		System.out.println(path + ": " + path + System.getProperty("file.separator") + args[1]);
		BufferedReader reader = new BufferedReader(new FileReader(path + System.getProperty("file.separator") + args[1]));
		ArffReader arff = new ArffReader(reader);
		Instances data = arff.getData();
//		System.out.println(data.numInstances());
		conf.setInt("testNumInstances", data.numInstances());
		
//		Instance inst;
//		while ((inst = arff.readInstance(data)) != null) {
//		   System.out.println(inst.attribute(0));
//		}
		
//		BufferedReader reader2 = new BufferedReader(new FileReader(path + System.getProperty("file.separator") + args[0]));
//		ArffReader arff2 = new ArffReader(reader2, 20);
//		Instances data2 = arff2.getData();
//		data2.setClassIndex(data2.numAttributes() - 1);
//		
//		AttributeStats as = data2.attributeStats(data2.numAttributes() - 1);
//		conf.setInt("numClasses", as.distinctCount);
//		System.out.println(as.distinctCount);
////		data = arff.getData();
//		Instance inst2;
//		while ((inst2 = arff2.readInstance(data2)) != null) {
//		   System.out.println(inst2.classValue());
//		}
		
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
		
		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}
	
}
