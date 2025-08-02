import sys
import os
import time
from pyspark.sql import SparkSession
from pyspark.sql import types as T
from pyspark.sql import functions as F
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utility.utility import setup_logging, format_time

def create_spark_session(logger, spark_config):
    """Initialize Spark session."""
    logger.info("Starting spark session")
    return (SparkSession.builder
            .master(f"spark://{spark_config['master_ip']}:7077")
            .appName("SpotifyDataTransform")
            .config("spark.driver.memory", spark_config["driver_memory"])
            .config("spark.executor.memory", spark_config["executor_memory"])
            .config("spark.executor.cores", spark_config["executor cores"]) 
            .config("spark.executor.instances", spark_config["executor_instances"]) 
            .getOrCreate()
    )

def load_and_clean(logger, spark, input_dir, output_dir):
    """Stage 1: Load data, drop duplicates, remove nulls, save cleaned data."""
    # Define schemas
    artists_schema = T.StructType([
        T.StructField("id", T.StringType(), False),
        T.StructField("followers", T.FloatType(), True),
        T.StructField("genres", T.StringType(), True),  
        T.StructField("name", T.StringType(), True),
        T.StructField("popularity", T.IntegerType(), True)
    ])
    
    recommendations_schema = T.StructType([
        T.StructField("id", T.StringType(), False),
        T.StructField("related_ids", T.ArrayType(T.StringType()), True)
    ])
    
    tracks_schema = T.StructType([
        T.StructField("id", T.StringType(), False),
        T.StructField("name", T.StringType(), True),
        T.StructField("popularity", T.IntegerType(), True),
        T.StructField("duration_ms", T.IntegerType(), True),
        T.StructField("explicit", T.IntegerType(), True),
        T.StructField("artists", T.StringType(), True),  
        T.StructField("id_artists", T.StringType(), True),
        T.StructField("release_date", T.StringType(), True),
        T.StructField("danceability", T.FloatType(), True),
        T.StructField("energy", T.FloatType(), True),
        T.StructField("key", T.IntegerType(), True),
        T.StructField("loudness", T.FloatType(), True),
        T.StructField("mode", T.IntegerType(), True),
        T.StructField("speechiness", T.FloatType(), True),
        T.StructField("acousticness", T.FloatType(), True),
        T.StructField("instrumentalness", T.FloatType(), True),
        T.StructField("liveness", T.FloatType(), True),
        T.StructField("valence", T.FloatType(), True),
        T.StructField("tempo", T.FloatType(), True),
        T.StructField("time_signature", T.IntegerType(), True)
    ])
    
    logger.info("Loading data in dataframes")
    artists_df = spark.read.schema(artists_schema).csv(os.path.join(input_dir, "artists.csv"), header=True)
    recommendations_df = spark.read.schema(recommendations_schema).json(os.path.join(input_dir, "fixed_da.json"))
    tracks_df = spark.read.schema(tracks_schema).csv(os.path.join(input_dir, "tracks.csv"), header=True)
    
    artists_df = artists_df.dropDuplicates(["id"]).filter(F.col("id").isNotNull())
    recommendations_df = recommendations_df.dropDuplicates(["id"]).filter(F.col("id").isNotNull())
    tracks_df = tracks_df.dropDuplicates(["id"]).filter(F.col("id").isNotNull())
    
    artists_df.write.mode("overwrite").parquet(os.path.join(output_dir, "stage1", "artists"))
    recommendations_df.write.mode("overwrite").parquet(os.path.join(output_dir, "stage1","recommendations"))
    tracks_df.write.mode("overwrite").parquet(os.path.join(output_dir, "stage1","tracks"))
    
    logger.info("Stage 1: Cleaned data saved")
    return artists_df, recommendations_df, tracks_df

def create_master_table(logger, output_dir, artists_df, recommendations_df, tracks_df):
    """Stage 2: Create master table by joining artists, tracks, and recommendations."""
   
    logger.info("Creation of master table started")
    tracks_df = tracks_df.withColumn("id_artists_array", F.from_json(F.col("id_artists"), F.ArrayType(F.StringType())))
    
    tracks_exploded = tracks_df.select("id", "name", "popularity", "id_artists_array").withColumn("artist_id", F.explode("id_artists_array"))
    
    # Join tracks with artists
    master_df = tracks_exploded.join(artists_df, tracks_exploded.artist_id == artists_df.id, "left") \
                              .select(
                                  tracks_exploded.id.alias("track_id"),
                                  tracks_exploded.name.alias("track_name"),
                                  tracks_exploded.popularity.alias("track_popularity"),
                                  artists_df.id.alias("artist_id"),
                                  artists_df.name.alias("artist_name"),
                                  artists_df.followers,
                                  artists_df.genres,
                                  artists_df.popularity.alias("artist_popularity")
                              )
    
    # Join with recommendations
    master_df = master_df.join(recommendations_df, master_df.artist_id == recommendations_df.id, "left") \
                         .select(
                             master_df.track_id,
                             master_df.track_name,
                             master_df.track_popularity,
                             master_df.artist_id,
                             master_df.artist_name,
                             master_df.followers,
                             master_df.genres,
                             master_df.artist_popularity,
                             recommendations_df.related_ids
                         )
    
    # Save master table
    master_df.write.mode("overwrite").parquet(os.path.join(output_dir, "stage2","master_table"))
    logger.info("Stage 2: Master table saved")
    

def create_query_tables(logger, output_dir, artists_df, recommendations_df, tracks_df):
    """Stage 3: Create query-optimized tables."""
    
    logger.info("Creation of individual tables started")
    recommendations_exploded = recommendations_df.withColumn("related_id", F.explode("related_ids")) \
                                                .select("id", "related_id")
    recommendations_exploded.write.mode("overwrite").parquet(os.path.join(output_dir, "stage3","recommendations_exploded"))
    
    tracks_exploded = tracks_df.withColumn("id_artists_array", F.from_json(F.col("id_artists"), T.ArrayType(T.StringType()))) \
                               .withColumn("artist_id", F.explode("id_artists_array")) \
                               .select("id", "artist_id")
    tracks_exploded.write.mode("overwrite").parquet(os.path.join(output_dir, "stage3","artist_track"))
    
    tracks_metadata = tracks_df.select(
        "id", "name", "popularity", "duration_ms", "danceability", "energy", "tempo"
    )
    tracks_metadata.write.mode("overwrite").parquet(os.path.join(output_dir, "stage3","track_metadata"))
    
    artists_metadata = artists_df.select("id", "name", "followers", "popularity")
    artists_metadata.write.mode("overwrite").parquet(os.path.join(output_dir, "stage3","artist_metadata"))
    
    logger.info("Stage 3: Query-optimized tables saved")

if __name__ == "__main__":

    logger = setup_logging("transform.log")
    
    if len(sys.argv) != 8:
        logger.critical("Usage: python3 transform/execute.py <input_dir> <output_dir> master_ip d_mem e_mem e_core e_inst")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    spark_config = {}
    spark_config["master_ip"] = sys.argv[3]
    spark_config["driver_memory"] = sys.argv[4]
    spark_config["executor_memory"] = sys.argv[5]
    spark_config["executor cores"] = sys.argv[6]
    spark_config["executor_instances"] = sys.argv[7]
    
    start = time.time()
    spark = create_spark_session(logger, spark_config)
    artists_df, recommendations_df, tracks_df = load_and_clean(logger, spark, input_dir, output_dir)
    create_master_table(logger, output_dir, artists_df, recommendations_df, tracks_df)
    create_query_tables(logger, output_dir, artists_df, recommendations_df, tracks_df)
    
    end = time.time()
    logger.info("Transformation pipeline completed")
    logger.info(f"Total time taken {format_time(end-start)}")
