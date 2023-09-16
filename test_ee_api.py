import ee
ee.Initialize()
print(ee.Image("NASA/NASADEM_HGT/001").get("title").getInfo())
