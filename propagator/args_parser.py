import argparse

def parse_params():
    parser = argparse.ArgumentParser(
        #prog='PROPAGATOR',
        description='''Run the PROPAGATOR model'''
    )

    parser.add_argument('-id', 
        dest='run_id', 
        type=str,  
        help='the run id'
    )
    parser.add_argument('-f', 
        dest='param_file', 
        type=argparse.FileType('r'), 
        help='parameter file for the model'
    )
    parser.add_argument('-of', 
        dest='output_folder', 
        type=str,   
        help='work folder'
    )
    parser.add_argument('-it', 
        dest='image_time', 
        type=str,       
        help='image timing'
    )
    parser.add_argument('-st', 
        dest='isochrone_time', 
        type=int, 
        help='isochrone timing'
    )
    parser.add_argument('-tl', 
        dest='time_limit', 
        type=int, 
        help='time limit in hours'
    )
    parser.add_argument('-d', 
        dest='debug_mode', 
        action='store_true', 
        help='debug mode', 
        default=False
    )
    parser.add_argument('-v',
        dest='write_vegetation',
        action='store_true', 
        help='write vegetation and dtm file to disk',
        default=False
    )    
    parser.add_argument('-sr', 
        dest='save_realizations', 
        action='store_true', 
        help='write a geotiff for each realization at the end of the simulation', 
        default=False
    )    
    parser.add_argument('-sf', 
        dest='simp_fact', 
        type=float, 
        default=0.00001, 
        help='simplify factor for isochrones'
    )

    parser.add_argument('-dem', 
        dest='dem_file', 
        type=argparse.FileType('r'),         
        help='D.E.M. tiff file to load'
    )

    parser.add_argument('-veg', 
        dest='veg_file', 
        type=argparse.FileType('r'),         
        help='Vegetation tiff file to load'
    )

    args = parser.parse_args()
    
    return args