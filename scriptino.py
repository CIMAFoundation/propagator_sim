import rasterio as rio
import numpy  as np
import random
import sys
import os
import itertools
import json
import subprocess as sp
#import pandas as pd
#import geopandas as gpd
#import xarray as xr



"""
with rio.open('veg.tif') as src:
	test_driver = src.driver
	test_height = src.height
	test_width = src.width
	test_count = src.count
	test_dtype= src.dtypes[0]
	test_crs = src.crs
	test_transform = src.transform
	test_bounds = src.bounds
	test_res  = src.res
	metadati_test  =  Tiff_metadata( test_driver, test_height, test_width, test_count, test_dtype, test_crs,test_transform, test_bounds)
	src.close()
"""




class Tiff_metadata:
	def __init__(self, test_driver,test_height,test_width, test_count, test_dtype, test_crs,test_transform,test_bounds, test_res):
		self.driver = test_driver
		self.height = test_height
		self.width = test_width
		self.count = test_count
		self.dtype = test_dtype
		self.crs = test_crs
		self.transform =  test_transform
		self.bounds = test_bounds 
		self.res = test_res


def copy_metadata_from_tiff(path):
	with rio.open(path) as src:
		test_driver = src.driver
		test_height = src.height
		test_width = src.width
		test_count = src.count
		test_dtype= src.dtypes[0]
		test_crs = src.crs
		test_transform = src.transform
		test_bounds = src.bounds
		test_res  = src.res
		metadati_test  =  Tiff_metadata( test_driver, test_height, test_width, test_count, test_dtype, test_crs,test_transform, test_bounds, test_res)
		src.close()	
	return metadati_test 





def create_slope_file(percentualeslope, metad, folder = ""):
	Z = np.zeros([metad.height, metad.width],  dtype =  metad.dtype)
	for  i in range(metad.height):
		for j in range(metad.width):
			Z[i,j] = metad.res[0] * percentualeslope * j
	temp_src = rio.open( folder+"slope"+str(percentualeslope)+".tif", 'w', driver = mdata.driver, height = mdata.height,  width = mdata.width, count  = mdata.count, dtype = mdata.dtype, crs = mdata.crs, transform = mdata.transform)
	temp_src.write(Z,1)
	temp_src.close()


"""
# I used veg.tif in order to create: 1)random vegetation 2)conifer vegetation 3)vertical splitting, upper rectangle broadleaves, lower rect. conifers

# randomveg.tif
#conifere_med.tif
#broadleaves_conifers.tif

The adopted metadata are...
mdata.height                                                                                                                         
Out[66]: 902
In [67]: mdata.width                                                                                                                          
Out[67]: 925
n [68]: mdata.res                                                                                                                            
Out[68]: (20.0, 20.0)
In [71]: mdata.bounds                                                                                                                         
Out[71]: BoundingBox(left=500212.7122, bottom=4915062.2855, right=518712.7122, top=4933102.2855)
In [72]: mdata.transform                                                                                                                      
Out[72]: 
Affine(20.0, 0.0, 500212.7122,
       0.0, -20.0, 4933102.2855)
In [74]: mdata.crs                                                                                                                            
Out[74]: CRS.from_epsg(32632)
"""

mdata = copy_metadata_from_tiff("./test/veg.tif")


"""
Uno studio di variazione deve includere: 
1) prova su differenti vegetazioni, random totale, uniforme conifere, e per ultimo meta' Latifoglie meta' conifere.
2) differenti pendenze, qui rappresentate, pendenze da 0 a  100%
3) differenti inclinazioni del vento 
4) differenti norme della velocita' del vento 
"""

#definisco le coordinate del punto medio del tif in ingresso
#lat_max = mdata.latitude.max()
#lat_min = mdata.latitude.min()
#lon_max = mdata.longitude.max()
#lon_min = mdata.longitude.min()
#lat_mid = (lat_max + lat_min) / 2
#lon_mid = (lon_max + lon_min) / 2


# definisco l'ipercubo dei parametri
#vegetations = [ "broadleaves_conifers.tif",]
vegetations = ["randomveg.tif", "conifere_med.tif", "broadleaves_conifers.tif"]
vegetations_labels = ["rand_veg", "conifer_veg", "broad_conifer_veg" ]

slopes = np.linspace(0,1,5)
slopes_label = [str(i)  for i in slopes]

#angles = np.linspace(0, np.pi, 4)
angles = [270,  240, 210, 180]
angles_label = ["0", "1o3pi", "2o3pi", "pi"]


speeds = [0.0,  30.0]  #km/h
speeds_label = [str(i)for i in speeds]

trials = list(itertools.product( list(range(len(vegetations))) , list(range(len(slopes))), list(range(len(angles))), list(range(len(speeds)))) )

#veg slope angle speed 
for  item  in trials:
	mylabel =    vegetations_labels[item[0]] +'_'+ slopes_label[item[1]] +'_'+ angles_label[item[2]] + '_' + speeds_label[item[3]] 

	if not os.path.exists("./test/"+mylabel):
		os.makedirs("./test/"+mylabel)

	#tmp = data["location"]

	with open("./test/base.json", "r") as jsonFile:
		data = json.load(jsonFile)
		jsonFile.close()

	data["boundary_conditions"][0]["w_speed"] = speeds[item[3]]
	data["boundary_conditions"][0]["w_dir"] = angles[item[2]]
	#data["boundary_conditions"][0]["ignitions"] = [str(lat_mid) , str(lon_mid)]

	with open("./test/"+mylabel + "/" + mylabel+".json", "w") as jsonFile:
		json.dump(data, jsonFile)

	create_slope_file(slopes[item[1]], mdata, "./test/slopes_test/" )
	commandlist =  ["python","main.py","-f","./test/"+mylabel+"/"+mylabel+".json"]
	commandlist.append("-o")
	commandlist.append("./test/"+mylabel)
	commandlist.append("-dem")
	commandlist.append("./test/slopes_test/slope"+slopes_label[item[1]]+".tif")
	commandlist.append( "-veg")
	commandlist.append("./test/vegetations_test/"+ vegetations[item[0]])

	commandliststring = "".join(ciao for ciao in commandlist)
	sp.run(commandlist)
	#sp.Popen(commandlist)
	# [latifoglie cespugli aree_nude erba conifere coltivi faggete]
