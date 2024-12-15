import gc
import json
import math
import tracemalloc
from matplotlib.pylab import uniform
import numpy as np
import psutil
from pymongo import MongoClient, ASCENDING, GEOSPHERE
import time
import matplotlib.pyplot as plt
import random

# Step-1: Setup MongoDB Connection
def setup_mongodb_connection(uri):
    try:
        client = MongoClient(uri)
        print("Connected to MongoDB successfully!")
        return client
    except Exception as e:
        print("Failed to connect to MongoDB:", e)
        return None

def read_geojson_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:  # Specify UTF-8 encoding
            data = json.load(file)
        print(f"Loaded {len(data)} records from the file.")
        return data
    except Exception as e:
        print("Error reading GeoJSON file:", e)
        return []

def drop_all_indexes(collection):
    try:
        collection.drop_indexes()  # Drops all indexes in the collection
        print("All indexes dropped successfully.")
    except Exception as e:
        print("Error dropping indexes:", e)

# https://geojson-maps.kyd.au/
#Format is correct for a GeoJSON file. It adheres to the GeoJSON specification, which commonly includes a FeatureCollection
# as the top-level type, and features as an array of Feature objects. Each feature contains properties for metadata and
# geometry for spatial data, which can represent points, lines, or polygons.
# However, MongoDB's geospatial queries typically require the geometry field to be directly at the document's root level
# (or a specific subfield, such as location), with a valid GeoJSON format for geospatial indexing.
def process_geojson1(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            geojson = json.load(file)

        # Extract features and prepare data for MongoDB insertion
        documents = []
        for feature in geojson.get("features", []):
            geometry = feature.get("geometry", {})
            properties = feature.get("geometry", {})
            latitude, longtitude = None, None
            coordinates_list = []
            coordinates = geometry.get("coordinates", [])

            if geometry.get("type") == "Point":
                if len(coordinates) >= 2:
                    coordinates_list.append({"longtitude": coordinates[0], "latitude": coordinates[1]})
            elif geometry.get("type") == "Polygon":
                for ring in coordinates:
                    for coord in ring:
                        if len(coord) >= 2:
                            coordinates_list.append({"longtitude": coord[0], "latitude": coord[1]})
            elif geometry.get("type") == "MultiPolygon":
                for polygon in coordinates:
                    for ring in polygon:
                        for coord in ring:
                            if len(coord) >= 2:
                                coordinates_list.append({"longtitude": coord[0], "latitude": coord[1]})
            document = {
                "properties": properties,
                "geometry": geometry,
                "coordinates_list": coordinates_list
            }
            documents.append(document)

        print(f"Processed {len(documents)} features.")
        return documents

    except Exception as e:
        print("Error processing GeoJSON file:", e)
        return []

def process_geojson(file_path):
    import json

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            geojson = json.load(file)

        # Extract features and prepare data for MongoDB insertion
        documents = []
        for feature in geojson.get("features", []):
            geometry = feature.get("geometry", {})
            properties = feature.get("properties", {})
            coordinates_list = []
            coordinates = geometry.get("coordinates", [])

            if geometry.get("type") == "Point":
                if len(coordinates) >= 2:
                    lng, lat = coordinates[0], coordinates[1]
                    if -180 <= lng <= 180 and -90 <= lat <= 90:
                        coordinates_list.append({"longitude": lng, "latitude": lat})
            elif geometry.get("type") == "Polygon":
                for ring in coordinates:
                    for coord in ring:
                        if len(coord) >= 2:
                            lng, lat = coord[0], coord[1]
                            if -180 <= lng <= 180 and -90 <= lat <= 90:
                                coordinates_list.append({"longitude": lng, "latitude": lat})
            elif geometry.get("type") == "MultiPolygon":
                for polygon in coordinates:
                    for ring in polygon:
                        for coord in ring:
                            if len(coord) >= 2:
                                lng, lat = coord[0], coord[1]
                                if -180 <= lng <= 180 and -90 <= lat <= 90:
                                    coordinates_list.append({"longitude": lng, "latitude": lat})
            document = {
                "properties": properties,
                "geometry": geometry,
                "coordinates_list": coordinates_list
            }
            documents.append(document)

        print(f"Processed {len(documents)} features.")
        return documents

    except Exception as e:
        print("Error processing GeoJSON file:", e)
        return []

def insert_geojson_data(collection, documents):
    if documents:
        try:
            result = collection.insert_many(documents)
            print(f"Inserted {len(result.inserted_ids)} documents.")
        except Exception as e:
            print("Failed to insert data:", e)
    else:
        print("No valid documents to insert.")

def measure_index_creation_time(collection):
    # Measure B-Tree Index Creation Time
    print("\nCreating B-Tree Index...")
    start_time = time.time()
    collection.create_index([("coordinates_list.latitude", ASCENDING),
                             ("coordinates_list.longtitude", ASCENDING)])
    btree_time = time.time() - start_time
    print(f"B-Tree Index created in {btree_time:.4f} seconds.")

    # Measure Geospatial 2dsphere Index Creation Time
    print("\nCreating Geospatial 2dsphere Index...")
    start_time = time.time()
    collection.create_index([("geometry", GEOSPHERE)])
    geo_time = time.time() - start_time
    print(f"Geospatial 2dsphere Index created in {geo_time:.4f} seconds.")

    return btree_time, geo_time

def measure_performance(func, collection, *args, times=10):
    """
    Measures total execution time, CPU usage, and peak memory usage for a function executed multiple times.
    Garbage collection is disabled for consistent measurements.
    """
    # Initialize accumulators for total measurements
    total_time = 0
    total_cpu = 0
    total_peak_memory = 0

    try:
        # Disable garbage collection for consistent memory measurements
        gc.disable()

        for i in range(times):
            #print(f"--- Run {i+1} ---")

            # Start monitoring memory with tracemalloc
            tracemalloc.start()

            # Record CPU usage before execution
            process = psutil.Process()
            cpu_before = process.cpu_percent(interval=None)

            # Execute the function and measure time
            start_time = time.time()
            result = func(collection, *args)  # Run the query
            elapsed_time = time.time() - start_time

            # Measure memory usage
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Measure CPU usage after execution
            cpu_after = process.cpu_percent(interval=None)
            cpu_usage = cpu_after - cpu_before

            # Convert peak memory to MB
            peak_memory_mb = peak_memory / 1024 / 1024

            # Accumulate total values
            total_time += elapsed_time
            total_cpu += cpu_usage
            total_peak_memory += peak_memory_mb

            # Print resource usage for each iteration
            # print(f"Elapsed Time: {elapsed_time:.4f} seconds")
            # print(f"CPU Usage: {cpu_usage:.2f}%")
            # print(f"Peak Memory Usage: {peak_memory_mb:.2f} MB\n")

        # Calculate averages
        avg_time = total_time / times
        avg_cpu = total_cpu / times
        avg_memory = total_peak_memory / times

        # Print cumulative results
        # print("\n--- Final Results After Multiple Runs ---")
        # print(f"Total Execution Time: {total_time:.4f} seconds")
        # print(f"Average Execution Time: {avg_time:.4f} seconds")
        # print(f"Average CPU Usage: {avg_cpu:.2f}%")
        # print(f"Average Peak Memory Usage: {avg_memory:.2f} MB\n")

        return result, total_time, avg_cpu, total_peak_memory

    except Exception as e:
        print("Error during resource monitoring:", e)
        return None, None, None, None
    finally:
        # Re-enable garbage collection
        gc.enable()

def measure_performance_old(query_function, collection, *args, times=10):
    total_time = 0
    total_cpu = 0
    total_mem = 0

    for i in range(times):
        process = psutil.Process()  # Track the current process stats

        # Start measuring
        cpu_start = process.cpu_percent(interval=None)  # CPU before query
        mem_start = process.memory_info().rss / (1024 * 1024)  # Memory before query (in MB)

        start_time = time.time()
        result = query_function(collection, *args)  # Run the query
        elapsed_time = time.time() - start_time

        # End measuring
        cpu_end = process.cpu_percent(interval=None)  # CPU after query
        mem_end = process.memory_info().rss / (1024 * 1024)  # Memory after query (in MB)

        # Accumulate metrics
        total_time += elapsed_time
        total_cpu += (cpu_end - cpu_start)
        total_mem += (mem_end - mem_start)

    # Calculate averages
    avg_time = total_time / times
    avg_cpu = total_cpu / times
    avg_mem = total_mem / times

    return result, total_time, total_cpu, total_mem

#################################### Point Query ##################################
# Point Query for B-Tree Index
def btree_point_query(collection, point):
    query = {
        "coordinates_list.latitude": point[1],
        "coordinates_list.longitude": point[0]
    }
    return list(collection.find(query))

# Point Query for Geospatial Index
def geospatial_point_query(collection, point):
    query = {
        "geometry": {
            "$geoIntersects": {
                "$geometry": {
                    "type": "Point",
                    "coordinates": point
                }
            }
        }
    }
    return list(collection.find(query))
#################################### End Point Query ##################################

####################################
# lat_min, lat_max = 9.5, 9.6
# lon_min, lon_max = -82.6, -82.5
# polygon = {
#     "type": "Polygon",
#     "coordinates": [[
#         [lon_min, lat_min],
#         [lon_max, lat_min],
#         [lon_max, lat_max],
#         [lon_min, lat_max],
#         [lon_min, lat_min] #close polygon in geospatial index
#     ]]
# }
# Range Query for B-Tree Index
def btree_range_query(collection, lat_min, lat_max, lon_min, lon_max):
    query = {
        "coordinates_list.latitude": {"$gte": lat_min, "$lte": lat_max},
        "coordinates_list.longitude": {"$gte": lon_min, "$lte": lon_max}
    }
    return list(collection.find(query))

# Geospatial Range Query
def geospatial_range_query(collection, polygon):
    query = {
        "geometry": {
            "$geoWithin": {
                "$geometry": polygon
            }
        }
    }
    return list(collection.find(query))
####################################
def btree_range_query_circle(collection, center, radius_in_km):
    import math

    def haversine_distance(lat1, lon1, lat2, lon2):
        # Haversine formula to calculate the distance between two points
        R = 6378.1  # Earth radius in kilometers
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    lat, lon = center
    lat_min = lat - (radius_in_km / 111)  # Approximation for 1 degree latitude = 111 km
    lat_max = lat + (radius_in_km / 111)
    lon_min = lon - (radius_in_km / (111 * math.cos(math.radians(lat))))
    lon_max = lon + (radius_in_km / (111 * math.cos(math.radians(lat))))

    # Query to fetch points in bounding box
    bounding_box_query = {
        "coordinates_list.latitude": {"$gte": lat_min, "$lte": lat_max},
        "coordinates_list.longitude": {"$gte": lon_min, "$lte": lon_max}
    }

    # Fetch results from MongoDB
    candidates = collection.find(bounding_box_query)

    # Refine the results to include only points within the circle
    results = []
    for candidate in candidates:
        for coord in candidate["coordinates_list"]:
            if haversine_distance(lat, lon, coord["latitude"], coord["longitude"]) <= radius_in_km:
                results.append(candidate)
                break  # No need to check other coordinates in this document

    return results

####################################################################################
def geospatial_range_query_circle(collection, center, radius_in_km):
    """
    Perform a range query using Geospatial index with a circular area.
    """
    query = {
        "geometry": {
            "$geoWithin": {
                "$centerSphere": [center, radius_in_km / 6378.1]  # Convert km to radians
            }
        }
    }
    return list(collection.find(query))
###################################################################################

#####################################################################################
# Point-in-Polygon B-Tree Query
def btree_point_in_polygon_query(collection, lat_min, lat_max, lon_min, lon_max, polygon):
    """
    Perform Point-in-Polygon query using B-Tree index.
    This filters by bounding box first, then verifies points inside the polygon.
    """
    def is_point_in_polygon(point, polygon_coords):
        """
        Check if a point is inside a polygon using ray-casting algorithm.
        """
        x, y = point
        n = len(polygon_coords)
        inside = False

        px, py = polygon_coords[0]
        for i in range(1, n + 1):
            nx, ny = polygon_coords[i % n]
            if y > min(py, ny):
                if y <= max(py, ny):
                    if x <= max(px, nx):
                        if py != ny:
                            xinters = (y - py) * (nx - px) / (ny - py) + px
                        if px == nx or x <= xinters:
                            inside = not inside
            px, py = nx, ny

        return inside
    # Get the bounding box coordinates from the polygon
    bounding_box_query = {
        "coordinates_list.latitude": {"$gte": lat_min, "$lte": lat_max},
        "coordinates_list.longtitude": {"$gte": lon_min, "$lte": lon_max}
    }
    results = collection.find(bounding_box_query)
    # Filter points by checking if they are inside the polygon
    polygon_coords = polygon["coordinates"][0]
    filtered_results = []
    for doc in results:
        for coord in doc["coordinates_list"]:
            point = [coord["longtitude"], coord["latitude"]]
            if is_point_in_polygon(point, polygon_coords):
                filtered_results.append(doc)
                break

    return filtered_results

# Point-in-Polygon Geospatial Query
def geospatial_point_in_polygon_query(collection, polygon):
    """
    Perform Point-in-Polygon query using Geospatial index.
    """
    query = {
        "geometry": {
            "$geoWithin": {
                "$geometry": polygon
            }
        }
    }
    # Perform geospatial query
    return list(collection.find(query))
##################################################################
import matplotlib.pyplot as plt

def plot_performance(index_creation_time, point_query_metrics, point_in_polygon_metrics, range_query_metrics):
    """
    Plot performance metrics: index creation time, point query, point-in-polygon, and range query.
    """
    # Unpack the index creation time (B-Tree, Geospatial)
    btree_index_time, geo_index_time = index_creation_time

    # Unpack the point query metrics (time, CPU, memory for B-Tree, Geospatial)
    time_btree_point, cpu_btree_point, mem_btree_point = point_query_metrics['B-Tree']
    time_geo_point, cpu_geo_point, mem_geo_point = point_query_metrics['Geospatial']

    # Unpack the point-in-polygon query metrics (time, CPU, memory for B-Tree, Geospatial)
    time_btree_pip, cpu_btree_pip, mem_btree_pip = point_in_polygon_metrics['B-Tree']
    time_geo_pip, cpu_geo_pip, mem_geo_pip = point_in_polygon_metrics['Geospatial']

    # Unpack the range query metrics (time, CPU, memory for B-Tree, Geospatial)
    time_btree_range, cpu_btree_range, mem_btree_range = range_query_metrics['B-Tree']
    time_geo_range, cpu_geo_range, mem_geo_range = range_query_metrics['Geospatial']

    # Create the figure and axes
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Part 1: Index Creation Time
    axs[0, 0].bar(['B-Tree', 'Geospatial'], [btree_index_time, geo_index_time], color=['blue', 'green'])
    axs[0, 0].set_title('Index Creation Time')
    axs[0, 0].set_ylabel('Time (seconds)')

    # Part 2: Point Query Metrics (Elapsed Time, CPU, Memory)
    axs[0, 1].bar(['B-Tree (Time)', 'Geospatial (Time)'], [time_btree_point, time_geo_point], color=['blue', 'green'])
    axs[0, 1].set_title('Point Query - Elapsed Time')
    axs[0, 1].set_ylabel('Time (seconds)')

    axs[1, 0].bar(['B-Tree (CPU)', 'Geospatial (CPU)'], [cpu_btree_point, cpu_geo_point], color=['blue', 'green'])
    axs[1, 0].set_title('Point Query - CPU Usage')
    axs[1, 0].set_ylabel('CPU (%)')

    axs[1, 1].bar(['B-Tree (Memory)', 'Geospatial (Memory)'], [mem_btree_point, mem_geo_point], color=['blue', 'green'])
    axs[1, 1].set_title('Point Query - Memory Usage')
    axs[1, 1].set_ylabel('Memory (MB)')

    # Part 3: Point-in-Polygon Metrics (Elapsed Time, CPU, Memory)
    fig, axs2 = plt.subplots(1, 3, figsize=(14, 5))

    axs2[0].bar(['B-Tree (Time)', 'Geospatial (Time)'], [time_btree_pip, time_geo_pip], color=['blue', 'green'])
    axs2[0].set_title('Point-in-Polygon Query - Elapsed Time')
    axs2[0].set_ylabel('Time (seconds)')

    axs2[1].bar(['B-Tree (CPU)', 'Geospatial (CPU)'], [cpu_btree_pip, cpu_geo_pip], color=['blue', 'green'])
    axs2[1].set_title('Point-in-Polygon Query - CPU Usage')
    axs2[1].set_ylabel('CPU (%)')

    axs2[2].bar(['B-Tree (Memory)', 'Geospatial (Memory)'], [mem_btree_pip, mem_geo_pip], color=['blue', 'green'])
    axs2[2].set_title('Point-in-Polygon Query - Memory Usage')
    axs2[2].set_ylabel('Memory (MB)')

    # Part 4: Range Query Metrics (Elapsed Time, CPU, Memory)
    fig, axs3 = plt.subplots(1, 3, figsize=(14, 5))

    axs3[0].bar(['B-Tree (Time)', 'Geospatial (Time)'], [time_btree_range, time_geo_range], color=['blue', 'green'])
    axs3[0].set_title('Range Query - Elapsed Time')
    axs3[0].set_ylabel('Time (seconds)')

    axs3[1].bar(['B-Tree (CPU)', 'Geospatial (CPU)'], [cpu_btree_range, cpu_geo_range], color=['blue', 'green'])
    axs3[1].set_title('Range Query - CPU Usage')
    axs3[1].set_ylabel('CPU (%)')

    axs3[2].bar(['B-Tree (Memory)', 'Geospatial (Memory)'], [mem_btree_range, mem_geo_range], color=['blue', 'green'])
    axs3[2].set_title('Range Query - Memory Usage')
    axs3[2].set_ylabel('Memory (MB)')

    # Display the plot
    plt.tight_layout()
    plt.show()

################################################################

# Main Function
if __name__ == "__main__":
    # MongoDB URI and Collection setup
    uri = "mongodb://localhost:27017"
    database_name = "spatial_query"
    collection_name = "spatial_collection"
    first = True
    client = setup_mongodb_connection(uri)
    if client:
        db = client[database_name]
        collection = db[collection_name]

        if first == True:
            geojson_file_path = "all.json"
            documents = process_geojson(geojson_file_path)
            insert_geojson_data(collection, documents)
        try:
            times = int(input("Enter the number of times to execute the queries: "))
        except ValueError:
            print("Invalid input! Defaulting to 1 query execution.")
            times = 1

         # Define index types to test
        index_types = ["B-Tree", "Hashed", "Geospatial"]
        creation_times = []
        elapsed_times = []
        cpu_usages = []
        mem_usages = []

        simple_times = []
        medium_times = []
        complex_times = []

        ################## Measure Index creating time #######################
        drop_all_indexes(collection)
        btree_time, geo_time = measure_index_creation_time(collection)
        print(f"\nSummary:")
        print(f"Time to create B-Tree Index: {btree_time:.4f} seconds")
        print(f"Time to create Geospatial 2dsphere Index: {geo_time:.4f} seconds")
        ################## End Measure Index creating time #######################

        latitude = 9.538818359375
        longitude = -82.58652343749999
        point = [longitude, latitude]
        range_lat_min, range_lat_max = 9.5, 9.6
        range_lon_min, range_lon_max = -82.6, -82.5
        range_polygon = {
            "type": "Polygon",
            "coordinates": [[
                [-82.6, 9.5],
                [-82.5, 9.5],
                [-82.5, 9.6],
                [-82.6, 9.6],
                [-82.6, 9.5]
            ]]
        }

        print("\n--- Point Query Performance ---")
        _, point_time_btree, point_cpu_btree, point_mem_btree = measure_performance(btree_point_query, collection, point, times = times)
        print(f"B-Tree Point Query - Time: {point_time_btree:.4f}s, CPU: {point_cpu_btree}%, Memory: {point_mem_btree:.2f}MB")

        _, point_time_geo, point_cpu_geo, point_mem_geo = measure_performance(geospatial_point_query, collection, point, times=times)
        print(f"Geospatial Point Query - Time: {point_time_geo:.4f}s, CPU: {point_cpu_geo}%, Memory: {point_mem_geo:.2f}MB")

        print("\n--- Range Query Performance ---")
        _, range_time_btree, range_cpu_btree, range_mem_btree = measure_performance(
            btree_range_query, collection, range_lat_min, range_lat_max, range_lon_min, range_lon_max)
        print(f"B-Tree Range Query - Time: {range_time_btree:.4f}s, CPU: {range_cpu_btree}%, Memory: {range_mem_btree:.2f}MB")

        _, range_time_geo, range_cpu_geo, range_mem_geo = measure_performance(
            geospatial_range_query, collection, range_polygon)
        print(f"Geospatial Range Query - Time: {range_time_geo:.4f}s, CPU: {range_cpu_geo}%, Memory: {range_mem_geo:.2f}MB")

        print("\n--- Point-in-Polygon Query Performance ---")
        _, poly_time_btree, poly_cpu_btree, poly_mem_btree = measure_performance(
            btree_point_in_polygon_query, collection, range_lat_min, range_lat_max, range_lon_min, range_lon_max, range_polygon)
        print(f"B-Tree Point-in-Polygon Query - Time: {poly_time_btree:.4f}s, CPU: {poly_cpu_btree}%, Memory: {poly_mem_btree:.2f}MB")

        _, poly_time_geo, poly_cpu_geo, poly_mem_geo = measure_performance(
            geospatial_point_in_polygon_query, collection, range_polygon)
        print(f"Geospatial Point-in-Polygon Query - Time: {poly_time_geo:.4f}s, CPU: {poly_cpu_geo}%, Memory: {poly_mem_geo:.2f}MB")

        # Example Usage
        # Assuming `measure_performance` gives metrics and `index_creation_time` has index creation times
        index_creation_time = (btree_time, geo_time)  # Example: Time to create B-Tree and Geospatial index
        point_query_metrics = {
            'B-Tree': [point_time_btree, point_cpu_btree, point_mem_btree],
            'Geospatial': [point_time_geo, point_cpu_geo, point_mem_geo]
        }
        range_query_metrics = {
            'B-Tree': [range_time_btree, range_cpu_btree, range_mem_btree],
            'Geospatial': [range_time_geo, range_cpu_geo, range_mem_geo]
        }
        point_in_polygon_metrics = {
            'B-Tree': [poly_time_btree, poly_cpu_btree, poly_mem_btree],
            'Geospatial': [poly_time_geo, poly_cpu_geo, poly_mem_geo]
        }

        plot_performance(index_creation_time, point_query_metrics, point_in_polygon_metrics, range_query_metrics)
