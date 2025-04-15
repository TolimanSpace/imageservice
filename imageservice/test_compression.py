import cProfile
import logging
import glob
from workers.images import compress_dump, compress_netcdf, compress_netcdf_bulk

logging.basicConfig(level=logging.DEBUG)

raw_files = glob.glob("images/raw/*.npy")

with cProfile.Profile() as pr:
    for raw_file in raw_files[:10]:
        # result = compress_dump(raw_file)
        result = compress_netcdf(raw_file)

    result = compress_netcdf_bulk(raw_files[:10], encoding={"zlib": True, "complevel": 9})

    pr.print_stats(sort='cumtime')