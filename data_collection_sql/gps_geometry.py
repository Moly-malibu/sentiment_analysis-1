import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


# Returns the vertices of the inscribed square
# R is in units of KM
def bounding_square(lon0,lat0,R):
    
    R = 1000.0*R # Convert to Meters
    
    xpt,ypt = m(lon0,lat0)
    
    x1,y1 = xpt-(R/np.sqrt(2.0)), ypt-(R/np.sqrt(2.0))
    x2,y2 = xpt+(R/np.sqrt(2.0)), ypt-(R/np.sqrt(2.0))
    x3,y3 = xpt-(R/np.sqrt(2.0)), ypt+(R/np.sqrt(2.0))
    x4,y4 = xpt+(R/np.sqrt(2.0)), ypt+(R/np.sqrt(2.0))
    
    x = [x1,x2,x3,x4]
    y = [y1,y2,y3,y4]
    
    
    return x,y

# Returns the vertices of the inscribed square
# R is units of KM
def bounding_square_coordinates(lon0,lat0,R):
    
    m = Basemap(projection='lcc', resolution='h', 
            lat_0=lat0, lon_0=lon0,
            width=1E6, height=1.2E6)
    
    R = 1000.0*R # Convert to Meters
    
    xpt,ypt = m(lon0,lat0)
    
    # Get the maximally distant vertices of the square
    x1,y1 = xpt-(R/np.sqrt(2.0)), ypt-(R/np.sqrt(2.0))
    x4,y4 = xpt+(R/np.sqrt(2.0)), ypt+(R/np.sqrt(2.0))
    
    # convert back to lat/lon
    lonpt1, latpt1 = m(x1,y1,inverse=True)
    lonpt4, latpt4 = m(x4,y4,inverse=True)
    
    return [lonpt1,latpt1,lonpt4,latpt4]
