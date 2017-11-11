import os
from star import makeTabbedLine, textHeader

def save_coords_in_star(starname,coords):
    # switch x,y
    coords = coords[:, ::-1]
    with open(starname, 'w') as starfile:
        starfile.write(textHeader(['CoordinateX', 'CoordinateY']))
        for coordX, coordY in coords:
            starfile.write(makeTabbedLine([coordX, coordY]))
