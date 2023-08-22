import numpy as np

def triangle_indices(b2, b1):
    #convert all the -1 in b2 to 1
    b2[b2==-1] = 1
    b1[b1==-1] = 1

    E, T = b2.shape
    N, _ = b1.shape

    triangles = []

    for t in range(T):
        # Get the edges involved in the current triangle
        edges_in_triangle = np.where(b2[:, t] == 1)[0]
       

        # Get the nodes involved in each edge
        nodes_in_edges = [np.where(b1[:, e] == 1)[0] for e in edges_in_triangle]
   

        # Flatten the list of nodes and remove duplicates
        nodes_in_triangle = np.unique(np.concatenate(nodes_in_edges))

        # Check if there are exactly 3 nodes in the triangle
        if len(nodes_in_triangle) == 3:
            triangles.append(nodes_in_triangle)

    if len(triangles) > 0:
        Tx3 = np.vstack(triangles)
    else:
        Tx3 = np.empty((0, 3), dtype=int)

    return Tx3
