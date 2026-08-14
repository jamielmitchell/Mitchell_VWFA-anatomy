import numpy as np
from scipy.stats import pearsonr, ttest_ind, chi2
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import pandas as pd


# define some usefull functions

def calculate_label_area(coords, faces, label_vertices):
    """Calculates surface area for a specific set of vertices."""
    
    # Get coordinates for all vertices
    v0 = coords[faces[:, 0]]
    v1 = coords[faces[:, 1]]
    v2 = coords[faces[:, 2]]
    
    # Calculate area of all triangles in the mesh
    face_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    
    # Create a mask for faces belonging to the label
    label_mask = np.zeros(len(coords), dtype=bool)
    label_mask[label_vertices] = True
    
    # A face is included if all its vertices are in the label
    face_in_label = label_mask[faces].all(axis=1)
    
    # Sum the area of faces within the label
    return np.sum(face_areas[face_in_label])


def annotate_regression_corr(data, x_param, y_param,fontsize=20, **kwargs):
    """Calculate Pearson's correlation coefficients and add them to plot

        Parameters :
        data : dataframe 
            data frame of data
        x_param : str
            name of df column plotted on x axis
        x_param : str
            name of df column plotted on y axis
        fontsize : int
            size of annotation font
        
    """
    # Remove rows with NaN values
    data = data.dropna(subset=[x_param, y_param])
    x = data[x_param]
    y = data[y_param]

    # calculate correlation coeficient
    r, p = pearsonr(x, y)


    if p < 0.001:
        significance = '***'
    elif p < 0.01:
        significance = '**'
    elif p < 0.05:
        significance = '*'
    elif p >= 0.05:
        significance = ''


    # Get axis object
    ax = plt.gca()

    # Annotate with coefficients and p-values
    ax.annotate(f'r: {r:.3f}\np: {p:.1e}', xy=(0.9, 0.05), xycoords='axes fraction', 
                fontsize=fontsize, ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0))


def load_label_file(label_file):
    """Load a label file and return the indices of the vertices."""
    vertices = []
    with open(label_file, 'r') as f:
        lines = f.readlines()
        for line in lines[2:]:  # Skip the first two lines (header)
            parts = line.strip().split()
            if len(parts) > 0:
                try:
                    vertex_index = int(parts[0])  # Convert to integer
                    vertices.append(vertex_index)
                except ValueError:
                    print(f"Skipping line due to ValueError: {line.strip()}")
    return np.array(vertices)

def get_surface_coordinates(mesh, vertices):
    """Extract the coordinates of the given vertices from the mesh."""
    # Use the coordinates of the mesh
    coords = mesh.coordinates[vertices]  # Extract relevant vertex coordinates
    return coords

def compute_pairwise_distances(coords):
    """Compute the pairwise distances between points."""
    return cdist(coords, coords)

def find_medoid(distances):
    """Find the medoid from a distance matrix."""
    sum_distances = distances.sum(axis=1)  # Sum distances from each point
    medoid_index = np.argmin(sum_distances)  # Index of the vertex minimizing sum distance
    return medoid_index

def compute_distance_to_medoid(coords, medoid_index):
    """Calculate distances from all vertices to the medoid vertex."""
    medoid_coord = coords[medoid_index].reshape(1, -1)  # Reshape for distance calculation
    distances = cdist(coords, medoid_coord)  # Get distances from all vertices to the medoid
    return distances.flatten()  # Flatten to a 1D array

def annotate_regression_corr(data, x_param, y_param,fontsize=20, **kwargs):
    """Calculate Pearson's correlation coefficients and add them to plot

        Parameters :
        data : dataframe 
            data frame of data
        x_param : str
            name of df column plotted on x axis
        x_param : str
            name of df column plotted on y axis
        fontsize : int
            size of annotation font
        
    """
    # Remove rows with NaN values
    data = data.dropna(subset=[x_param, y_param])
    x = data[x_param]
    y = data[y_param]

    # calculate correlation coeficient
    r, p = pearsonr(x, y)


    if p < 0.001:
        significance = '***'
    elif p < 0.01:
        significance = '**'
    elif p < 0.05:
        significance = '*'
    elif p >= 0.05:
        significance = ''


    # Get axis object
    ax = plt.gca()

    # Annotate with coefficients and p-values
    ax.annotate(f'r: {r:.3f}\np: {p:.3f}', xy=(0.9, 0.05), xycoords='axes fraction', 
                fontsize=fontsize, ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0))

# Create a combined color mapping column
def get_color(row,vwfa1_palette,vwfa2_palette):
    if row['roi'] == 'VWFA1':
        return vwfa1_palette[row['dys_group']]
    elif row['roi'] == 'VWFA2':
        return vwfa2_palette[row['dys_group']]
    else:
        return dysGroup_palette[row['dys_group']]  # fallback
    
    

def confidence_ellipsoid(data, confidence=0.95):
    """
    Calculate the confidence ellipsoid for 3D data.
    
    Parameters:
    -----------
    data : array-like, shape (n_samples, 3)
        The 3D data points
    confidence : float
        Confidence level (default: 0.95 for 95% confidence)
    
    Returns:
    --------
    center : array, shape (3,)
        Center of the ellipsoid
    radii : array, shape (3,)
        Radii of the ellipsoid along principal axes
    rotation : array, shape (3, 3)
        Rotation matrix (eigenvectors)
    """
    # Calculate mean (center of ellipsoid)
    center = np.mean(data, axis=0)
    
    # Calculate covariance matrix
    cov = np.cov(data.T)
    
    # Get eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Calculate chi-square value for confidence level
    # For 3D, degrees of freedom = 3
    chi2_val = chi2.ppf(confidence, df=3)
    
    # Calculate radii
    radii = np.sqrt(chi2_val * eigenvalues)
    
    return center, radii, eigenvectors

def plot_ellipsoid(ax, center, radii, rotation, color='blue', alpha=0.2, label=None):
    """
    Plot a 3D ellipsoid.
    
    Parameters:
    -----------
    ax : matplotlib 3D axis
        The axis to plot on
    center : array, shape (3,)
        Center of the ellipsoid
    radii : array, shape (3,)
        Radii along principal axes
    rotation : array, shape (3, 3)
        Rotation matrix
    color : str
        Color of the ellipsoid
    alpha : float
        Transparency level
    label : str
        Label for legend
    """
    # Create mesh grid
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    
    # Create unit sphere
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    
    # Stack coordinates
    sphere = np.stack([x, y, z], axis=-1)
    
    # Scale by radii and rotate
    ellipsoid = sphere @ np.diag(radii) @ rotation.T
    
    # Translate to center
    ellipsoid += center
    
    # Plot surface
    ax.plot_surface(ellipsoid[..., 0], ellipsoid[..., 1], ellipsoid[..., 2],
                    color=color, alpha=alpha, label=label, shade=True)

# Main plotting code for your data
def plot_vwfa_data(df, color_map, label_map, scatter_alpha = 0.5, ellipsoid_alpha=0.1, confidence=0.95, figsize=(12, 9)):
    """
    Create a 3D scatter plot with confidence ellipsoids for dyslexic vs typical groups.
    
    Parameters:
    -----------
    df_vwfa : pandas DataFrame
        Your dataframe with columns: x, y, z, sub, dys_group
    confidence : float
        Confidence level for ellipsoids (default: 0.95)
    figsize : tuple
        Figure size
    """
    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    
    # Get unique groups (should be 'dys' and 'typ')
    groups = df['dys_group'].unique()
    
    # Plot each group
    for group in groups:
        # Filter data for this group
        mask = df['dys_group'] == group
        data = df[mask][['x', 'y', 'z']].values
        
        # Get color for this group
        color = color_map.get(group, '#95A5A6')  # Default gray if group not in map
        label = label_map.get(group, group)
        
        # Plot scatter points
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], 
                  c=color, label=label,
                  s=50, alpha=scatter_alpha, edgecolors='k', linewidth=0.5)
        
        # Calculate and plot confidence ellipsoid
        center, radii, rotation = confidence_ellipsoid(data, confidence)
        plot_ellipsoid(ax, center, radii, rotation, 
                      color=color, alpha=ellipsoid_alpha)
    
    # Set labels
    ax.set_xlabel('X Coordinate', fontsize=12, labelpad=10)
    ax.set_ylabel('Y Coordinate', fontsize=12, labelpad=10)
    ax.set_zlabel('Z Coordinate', fontsize=12, labelpad=50)
    
    # Set title
    ax.set_title(f'VWFA Coordinates: Dyslexic vs Typical\n({int(confidence*100)}% Confidence Ellipsoids)', 
                 fontsize=14, pad=20)
    
    # Add legend
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    
    # Improve viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


