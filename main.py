#!/usr/bin/env python3
# Written by Yves OESCH / HB9DTX / http://www.yvesoesch.ch
# Project hosted on https://github.com/HB9DTX/OsmMarker
# Documentation in README.md
#
# 2022_10_25_Initial version

import pandas as pd  # sudo apt-get install python3-pandas or `sudo pip install pandas` to reduce 500MB apt(8) download
                    # requires also openpyxl as dependency to access .xlsx files
import logging
import matplotlib.pyplot as plt
import geotiler     # as `pip` package, usage: https://wrobell.dcmod.org/geotiler/usage.html

#logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
logging.info('Program START')

MARKERFILE = 'markers.xlsx'


df = pd.read_excel(MARKERFILE, sheet_name='Points')
logging.debug(df)
logging.debug(df['Name'][1])
logging.debug(df['Longitude'])
logging.debug(df['Latitude'])

Settings = dict(pd.read_excel(MARKERFILE, sheet_name='Settings',  header=None).values)

logging.debug(Settings)
logging.debug(Settings['MinLon'])
logging.debug(Settings['Title'])

MAP_BBOX = (Settings['MinLon'], Settings['MinLat'], Settings['MaxLon'], Settings['MaxLat'])  # Map limits; lower left, upper right, (long, lat) # central europe

mm = geotiler.Map(extent=MAP_BBOX, zoom=Settings['Zoom'])
img = geotiler.render_map(mm)

# points = list(zip(df['Longitude'], df['Latitude']))
# x, y = zip(*(mm.rev_geocode(p) for p in points))
# logging.debug(x)
# logging.debug(y)


fig3, ax3 = plt.subplots(tight_layout=True)
ax3.imshow(img)
for ind in df.index:
    logging.debug(df['Longitude'][ind])
    logging.debug(df['Latitude'][ind])
    x, y = mm.rev_geocode([df['Longitude'][ind], df['Latitude'][ind]])
    ax3.scatter(x, y, c=df['Color'][ind], edgecolor=df['Color'][ind], s=df['Size'][ind], alpha=0.9, label='all stations')
ax3.set_title(Settings['Title'])
plt.axis('off')
#annotation = 'Total number of stations in log: ' + str(contest.qsoList.shape[0])
#ax3.text(1, 0, annotation, transform=ax3.transAxes, ha='right', va='bottom',
#         bbox=dict(boxstyle='square', facecolor='white'))

annotation = Settings['Annotation']
if annotation != '':
    ax3.annotate(annotation, xy=(1, 0), xycoords='axes fraction', fontsize='xx-small', color='black', transform=ax3.transAxes, ha='right', va='bottom')
    #ax3.legend(annotation)
annotation = 'Plot: OsmMarker by Ynovo\nMap data: OpenStreetMap.'
ax3.text(0, 0, annotation, fontsize='xx-small', color='blue', transform=ax3.transAxes, ha='left', va='bottom')
# bbox=dict(boxstyle='square', facecolor='white'))
#    plt.savefig('Map.png', bbox_inches='tight')
plt.show()
plt.close()

logging.info('Program END')