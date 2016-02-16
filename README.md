# Morar

Storing CellProfiler data

## Purpose

- Create a database from .csv files produced by CellProfiler
- Relate objects within the same image to one another
- Relate child objects (such as nucleoli) to parent objects (cells)
- Want to ability to perform some pre-processing steps on the database
  without the need to move large datasets back into memory
- Link images to objects via image number and X-Y co-ordinates
- Open images using bioformats or similar image viewing library
