""" 
    1. decide on value for k. 
		a. has to be more than number of classes in our data
		b. must be at most the number of training data
	2. init the k cluster centers (randomly, if necessary)
		a. compute max and min for each dimension. randomly sample k times within that range
	3. decide the class memberships of the N training objects by assigning each to nearest cluster center
	4. Re-estimate k cluster centers, by assuming memberships found in last step are correct
		a. take the actual center of the points found to be members of the cluster
If none of the N objects changed membership in the last iteration, exit. Otherwise loop back to step 3.
"""

