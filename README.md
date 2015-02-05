# regional_names
This project uses Twitter data to classify individual tweets as originating in one of several arbitrarily defined US regions
using text classification methods and modules in Python. Most tweets include geocoordinates, which are used in this project
to determine training and test target labels. A series of mutually exclusive bounding boxes that encompass the contiguous
continental United States are the arbitrarily defined regions - currently there are 6 such regions:

West is: CA, OR, WA, ID, NV, AZ, UT, MT
west=[-124.78366, 24.545905,-110.897437, 48.997502]

Lowerwest is: 
lowerwest=[-110.897437, 24.545905, -97.274391, 39.881000]

Upperwest is:
upperwest=[-110.897437, 39.881000, -97.274391, 48.997502]

Midwest is:
midwest=[-97.274391, 39.475135, -80.926735, 48.997502]

South is:
south=[-97.274391, 24.545905, -67.097198, 39.475135]

Eastcoast is:
eastcoast=[-80.926735, 39.475135, -67.097198, 48.997502]
