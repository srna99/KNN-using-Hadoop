import weka.core.Instances;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class Hadoop {
	
	public static class KNNMapper extends Mapper {
		
	}
	
	public static class KNNReducer extends Reducer {
		
	}

	public static void main(String[] args) throws Exception {
		Configuration conf = new Configuration();
		
		Job job = Job.getInstance(conf, "knn");
		job.setJarByClass(Hadoop.class);
		job.setMapperClass(KNNMapper.class);
		job.setCombinerClass(KNNReducer.class);
		job.setReducerClass(KNNReducer.class);
//		job.setOutputKeyClass();
//		job.setOutputValueClass();
		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[1]));
		
		System.exit(job.waitForCompletion(true) ? 0 : 1);
	}
	
}
