#!/usr/bin/env python3
# Written by Yves OESCH / HB9DTX / http://www.yvesoesch.ch
# Project hosted on https://github.com/HB9DTX/OsmMarker
# Documentation in README.md
#
# 2022_10_25_Initial version
# Refer to git commits for history

import pandas as pd
# if not available: sudo apt-get install python3-pandas or `sudo pip install pandas` to reduce 500MB apt(8) download
# requires also openpyxl as dependency to access .xlsx files
import logging
import matplotlib.pyplot as plt
import geotiler     # as `pip` package, usage: https://wrobell.dcmod.org/geotiler/usage.html

logging.basicConfig(level=logging.DEBUG)
# logging.basicConfig(level=logging.INFO)
logging.info('Program START')

MARKERFILE = 'markers.xlsx'     # File containing markers info (coordinates, color,...)
MAPFILE = 'Map.png'

points = pd.read_excel(MARKERFILE, sheet_name='Points')
logging.debug(points)
logging.debug(points['Name'][1])
logging.debug(points['Longitude'])
logging.debug(points['Latitude'])

settings = dict(pd.read_excel(MARKERFILE, sheet_name='Settings',  header=None).values)

logging.debug(settings)
logging.debug(settings['MinLon'])

annotations = pd.read_excel(MARKERFILE, sheet_name='Annotations', dtype=str)
logging.debug(annotations)


MAP_BBOX = (settings['MinLon'], settings['MinLat'], settings['MaxLon'], settings['MaxLat'])
# Map limits; lower left, upper right, (long, lat) # central europe

logging.getLogger().setLevel(logging.INFO)
mm = geotiler.Map(extent=MAP_BBOX, zoom=settings['Zoom'])
img = geotiler.render_map(mm)

fig, ax = plt.subplots(tight_layout=True)
ax.imshow(img)
for ind in points.index:                        # plot each point on the map
    logging.debug(points['Longitude'][ind])
    logging.debug(points['Latitude'][ind])
    x, y = mm.rev_geocode([points['Longitude'][ind], points['Latitude'][ind]])
    ax.scatter(x, y, c=points['Color'][ind].lower(), edgecolor=points['Color'][ind].lower(),
               s=points['Size'][ind], marker=points['MarkerStyle'][ind],
               alpha=0.9)

plt.axis('off')
logging.getLogger().setLevel(logging.INFO)
for ind in annotations.index:                   # plot each annotation
    logging.getLogger().setLevel(logging.DEBUG)
    logging.debug(annotations['Name'][ind])
    logging.debug(annotations['Text'][ind])
    logging.debug(annotations['Color'][ind])
    logging.debug(annotations['Size'][ind])
    logging.getLogger().setLevel(logging.INFO)
    annotation = ''         # Needed if an annotation is empty, otherwise former annotation is re-used
    if annotations['Text'][ind] == annotations['Text'][ind]:
        annotation = bytes(annotations['Text'][ind], "utf-8").decode("unicode_escape")
        # needed to evaluate \n characters as new lines and not as string literal
        # ref: https://stackoverflow.com/questions/4020539/process-escape-sequences-in-a-string-in-python

    match annotations['Name'][ind]:
        case 'Title':
            ax.set_title(annotation,
                         color=annotations['Color'][ind].lower(),
                         fontsize=annotations['Size'][ind].lower())

        case 'TopLeft':
            if annotation == annotation:    # test if not NaN (empty Excel cell)
                ax.text(0, 0.98, annotation,
                        transform=ax.transAxes, ha='left', va='top',
                        color=annotations['Color'][ind].lower(),
                        fontsize=annotations['Size'][ind].lower(),
                        bbox=dict(boxstyle='square', facecolor='white', linewidth=0))
        case 'TopRight':
            if annotation == annotation:    # test if not NaN (empty Excel cell)
                ax.text(1, 0.98, annotation,
                        transform=ax.transAxes, ha='right', va='top',
                        color=annotations['Color'][ind].lower(),
                        fontsize=annotations['Size'][ind].lower(),
                        bbox=dict(boxstyle='square', facecolor='white', linewidth=0))
        case 'BottomLeft':
            if annotation == annotation:    # test if not NaN (empty Excel cell)
                ax.text(0, 0, annotation,
                        transform=ax.transAxes, ha='left', va='bottom',
                        color=annotations['Color'][ind].lower(),
                        fontsize=annotations['Size'][ind].lower(),
                        bbox=dict(boxstyle='square', facecolor='white', linewidth=0))
        case 'BottomRight':
            if annotation == annotation:    # test if not NaN (empty Excel cell)
                ax.text(1, 0, annotation,
                        transform=ax.transAxes, ha='right', va='bottom',
                        color=annotations['Color'][ind].lower(),
                        fontsize=annotations['Size'][ind].lower(),
                        bbox=dict(boxstyle='square', facecolor='white', linewidth=0))

plt.savefig(MAPFILE, bbox_inches='tight')
plt.show()
plt.close()

logging.info('Program END')
