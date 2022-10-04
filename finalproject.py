import numpy as np
import random
import matplotlib.pyplot as plt


"""
Mauricio Tedeschi
CS2810 - Mathematics of Data Models
14 December 2021

k-means clustering implementation with Jaccard score analysis
"""
    
class Point:
    def __init__(self, pos, clusterID):
        self.pos = pos
        self.clusterID = clusterID
        
    def get_dist(self, other):
        r = self.pos - other.pos
        return np.sqrt(np.dot(r, r))
    
    def get_ID_from_nearest_neighbor(self, M):
        min_d = np.Inf
        for i in range(len(M)):
            d = self.get_dist(M[i])
            if d < min_d:
                min_d = d
                self.clusterID = i
    
    def to_string(self):
        return "((" + str(self.pos) + ", " + str(self.clusterID) + ")"
            
# dataset is a list of (x1, x2, ..., x_N), the number of dimensions 
# is number of dimensions in every vector element

# returns list of points labeled with a cluster ID: C = 0...(k-1)
def k_means(file_data, title_args, k):
    dataset = read_data_from_file(file_data, title_args)
    means = random.sample(dataset, k)
    previous_means = []
    while previous_means != means:
        previous_means = means
        for x in dataset:
            x.get_ID_from_nearest_neighbor(means)
        clusters = [[x for x in dataset if x.clusterID == i] for i in range(k)]
        for i in range(k):
            new_centroid_pos = sum([x.pos for x in clusters[i]]) / len(clusters[i])
            means[i] = Point(new_centroid_pos, i)
    return clusters
    
# title_args is the list of column titles for the parameters 
# of interest
def read_data_from_file(filename, title_args):
    file = open(filename)
    is_header = True
    column_headers = []
    output_list = []
    for line in file:
        element = np.array([])
        line_sep = line.split(",")
        if is_header:
            column_headers = line_sep
            is_header = False
        else:
            for i in range(len(line_sep)):
                if column_headers[i] in title_args:
                    element = np.append(element, float(line_sep[i]))
            output_list.append(Point(element, -1))
    return output_list

def get_jaccard(l1, l2):
    s1 = set([tuple(x.pos) for x in l1])
    s2 = set([tuple(x.pos) for x in l2])
    return len(s1 & s2) / len(s1 | s2)

# Although k-means can work around any number of dimensions, the plotter 
# will only display the first two
def display_k_means_plot(dataset, k):
    color_modes = ["r.", "g.", "b.", "m.", "k.", "y."]
    for i in range(min(len(color_modes), k)):
        plt.plot([x.pos[0] for x in dataset[i]], [x.pos[1] for x in dataset[i]], color_modes[i])
        
def k_means_analysis(file, title_args, k):
    output = k_means(file, title_args, k)
    other_out = k_means(file, title_args, k)
    
    for i in range(k):
        for j in range(k):
            print("Cluster", str(i), "with", "Cluster", str(j), ":", get_jaccard(output[i], other_out[j]))
    print()
    
    display_k_means_plot(output, k)
    plt.show()
    display_k_means_plot(other_out, k)
    plt.show()
    
if __name__ == "__main__":
    k_means_analysis("housing.csv", ["Longitude", "Latitude"], 4)
    k_means_analysis("abalone.csv", ["Diameter", "Height"], 4)
    k_means_analysis("autos.csv", ["width", "height"], 4)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    