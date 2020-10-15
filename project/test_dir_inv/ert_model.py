# import pygimli as pg
# import pybert as pb
# import numpy as np
# import matplotlib.pyplot as plt
# from pygimli import meshtools as mt
# from pygimli.physics import ert
#
# world = mt.createWorld(start=[-5,0],end=[15,-4],layers=[-1.5,-3.2])
#
# #ДЕЛАЕМ КРИВУЮ ДЛЯ CREATE_POLYGON
# x = np.linspace(-5,15,201)
# y = -2 - np.exp(-(x-6.5)**2)
# curve = np.transpose(np.vstack((x,y)))
#
# # НАТЫКАЛИ ВСЯКОГО НЕОДНОРОДНОГО
# line1 = mt.createPolygon(curve,isClosed=False)
# line2 = mt.createLine(start=[-5,-0.2],end=[15,-1.3])
# circle = mt.createCircle(pos=[4,-0.4],radius=[0.5,0.2],marker=4)
# block = mt.createRectangle(start=[6,-0.1],end=[7.5,-0.5],marker=5)
#
# # СОЗДАЛИ МОДЕЛЬ И ПОКАЗАЛИ ЕЕ
# model = world + line1 + line2 + circle + block
# # pg.show(model)
# print(type(model))
#
# # СХЕМА ДИПОЛЬ-ДИПОЛЬ, расст. м/у эл. = 100 м
# scheme = ert.createERTData(elecs=np.linspace(start=0,stop=10,num=41),schemeName='slm')
#
# # СДЕЛАЛИ ДИСКРЕТИЗАЦИЮ БУДУЩЕЙ СЕТКИ КАК 10% РАССТОЯНИЯ М/У ЭЛЕКТРОДАМИ, ДЛЯ НОРМ КАЧЕСТВА
# for p in scheme.sensors():
#     model.createNode(p)
#     model.createNode(p - [0, 0.1])
#
# # ЗАПИЛИЛИ СЕТКУ:
# mesh = mt.createMesh(model, quality=34)
# # ДАТА ДЛЯ СЕТКИ:
# rhomap = [[0,75],[1,50],[2,125],[3,200],[4,200],[5,300]]
# # ВОТ КАК ОНА ВЫГЛЯДИТ:
# # pg.show(mesh,data=rhomap,label=pg.unit('res'),showMesh=True)
# print(type(mesh))
# mesh.save('mesh_test')

import matplotlib.pyplot as plt
import numpy as np

import pygimli as pg
# pg.setTestingMode(True)
import pygimli.meshtools as mt
import pygimli.physics.ert as ert


def plot_ert():
    # -*- coding: utf-8 -*-
    """
    2D ERT modeling and inversion
    -----------------------------
    """
    # sphinx_gallery_thumbnail_number = 6

    ###############################################################################
    # Create geometry definition for the modelling domain.
    #
    # worldMarker=True indicates the default boundary conditions for the ERT
    world = mt.createWorld(start=[-50, 0], end=[50, -50], layers=[-1, -5],
                           worldMarker=True)

    ###############################################################################
    # Create some heterogeneous circular anomaly
    block = mt.createCircle(pos=[-5, -3.], radius=[4, 1], marker=4,
                            boundaryMarker=10, area=0.1)

    ###############################################################################
    poly = mt.createPolygon([(1,-4), (2,-1.5), (4,-2), (5,-2),
                             (8,-3), (5,-3.5), (3,-4.5)], isClosed=True,
                             addNodes=3, interpolate='spline', marker=5)

    ###############################################################################
    # Merge geometry definition into a Piecewise Linear Complex (PLC)
    geom = world + block + poly

    ###############################################################################
    # Optional: show the geometry
    pg.show(geom)

    ###############################################################################
    # Create a Dipole Dipole ('dd') measuring scheme with 21 electrodes.
    scheme = ert.createERTData(elecs=np.linspace(start=-15, stop=15, num=9),
                               schemeName='dd'
                               )
    return
    ###############################################################################
    # Put all electrode (aka sensors) positions into the PLC to enforce mesh
    # refinement. Due to experience, its convenient to add further refinement
    # nodes in a distance of 10% of electrode spacing to achieve sufficient
    # numerical accuracy.
    for p in scheme.sensors():
        geom.createNode(p)
        geom.createNode(p - [0, 0.1])

    # Create a mesh for the finite element modelling with appropriate mesh quality.
    mesh = mt.createMesh(geom, quality=34)

    # Create a map to set resistivity values in the appropriate regions
    # [[regionNumber, resistivity], [regionNumber, resistivity], [...]
    rhomap = [[1, 100.],
              [2, 75.],
              [3, 50.],
              [4, 150.],
              [5, 25]]

    # Take a look at the mesh and the resistivity distribution
    pg.show(mesh, data=rhomap, label=pg.unit('res'), showMesh=True)

    # %%

    ###############################################################################
    # Perform the modeling with the mesh and the measuring scheme itself
    # and return a data container with apparent resistivity values,
    # geometric factors and estimated data errors specified by the noise setting.
    # The noise is also added to the data. Here 1% plus 1µV.
    # Note, we force a specific noise seed as we want reproducable results for
    # testing purposes.
    data = ert.simulate(mesh, scheme=scheme, res=rhomap, noiseLevel=1,
                        noiseAbs=1e-6, seed=1337)

    pg.warning(np.linalg.norm(data['err']), np.linalg.norm(data['rhoa']))
    pg.info('Simulated data', data)
    pg.info('The data contains:', data.dataMap().keys())

    pg.info('Simulated rhoa (min/max)', min(data['rhoa']), max(data['rhoa']))
    pg.info('Selected data noise %(min/max)', min(data['err'])*100, max(data['err'])*100)

    ###############################################################################
    # Optional: you can filter all values and tokens in the data container.
    # Its possible that there are some negative data values due to noise and
    # huge geometric factors. So we need to remove them.
    data.remove(data['rhoa'] < 0)
    pg.info('Filtered rhoa (min/max)', min(data['rhoa']), max(data['rhoa']))

    # You can save the data for further use
    data.save('simple.dat')

    # You can take a look at the data
    ert.show(data)
    ###############################################################################
    # Initialize the ERTManager, e.g. with a data container or a filename.
    #
    mgr = ert.ERTManager('simple.dat')
    ###############################################################################
    # Run the inversion with the preset data. The Inversion mesh will be created
    # with default settings.
    inv = mgr.invert(lam=20, verbose=True)
    # np.testing.assert_approx_equal(mgr.inv.chi2(), 0.6883, significant=1)

    ###############################################################################
    # Let the ERTManger show you the model of the last successful run and how it
    # fits the data. Shows data, model response, and model.
    #
    mgr.showResultAndFit()
    return
    meshPD = pg.Mesh(mgr.paraDomain) # Save copy of para mesh for plotting later
    # %%
    # You can also provide your own mesh (e.g., a structured grid if you like them)
    #
    inversionDomain = pg.createGrid(x=np.linspace(start=-18, stop=18, num=33),
                                    y=-pg.cat([0], pg.utils.grange(0.5, 8, n=8)),
                                    marker=2)
    ###############################################################################
    # The inversion domain for ERT problems needs a boundary that represents the
    # far regions in the subsurface of the halfspace.
    # Give a cell marker lower than the marker for the inversion region, the lowest
    # cell marker in the mesh will be the inversion boundary region by default.
    #
    grid = pg.meshtools.appendTriangleBoundary(inversionDomain, marker=1,
                                               xbound=50, ybound=50)

    ###############################################################################
    # The Inversion can be called with data and mesh as argument as well
    #
    model = mgr.invert(data, mesh=grid, lam=20, verbose=True)
    # np.testing.assert_approx_equal(mgr.inv.chi2(), 0.951027, significant=3)
    ###############################################################################
    # You can of course get access to mesh and model and plot them for your own.
    # Note that the cells of the parametric domain of your mesh might be in
    # a different order than the values in the model array if regions are used.
    # The manager can help to permutate them into the right order.
    #
    modelPD = mgr.paraModel(model)  # do the mapping
    pg.show(mgr.paraDomain, modelPD, label='Model', cMap='Spectral_r',
            logScale=True, cMin=25, cMax=150)

    pg.info('Inversion stopped with chi² = {0:.3}'.format(mgr.fw.chi2()))

    fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True, sharey=True, figsize=(8,7))

    pg.show(mesh, rhomap, ax=ax1, hold=True, cMap="Spectral_r", logScale=True,
            orientation="vertical", cMin=25, cMax=150)
    pg.show(meshPD, inv, ax=ax2, hold=True, cMap="Spectral_r", logScale=True,
            orientation="vertical", cMin=25, cMax=150)
    mgr.showResult(ax=ax3, cMin=25, cMax=150, orientation="vertical")

    labels = ["True model", "Inversion unstructured mesh", "Inversion regular grid"]
    for ax, label in zip([ax1, ax2, ax3], labels):
        ax.set_xlim(mgr.paraDomain.xmin(), mgr.paraDomain.xmax())
        ax.set_ylim(mgr.paraDomain.ymin(), mgr.paraDomain.ymax())
        ax.set_title(label)


plot_ert()

plt.show()
