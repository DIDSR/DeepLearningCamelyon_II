# Install openslide

sudo apt-get install openslide-tools
pip install Pillow
pip install openslide-python


# Use openslide after entering python3 shell

slide = openslide.open_slide(slide_path)

thumbnail = slide.get_thumbnail((slide.dimensions[0] / 256, slide.dimensions[1] / 256))


# Display WSI on web browser (python package flask is required)

git clone https://github.com/openslide/openslide-python.git


python3 path/to/openslide-python/examples/deepzoom/deepzoom_multiserver.py -Q 100 path/to/data/ 


## For example:

python3 openslide-python/examples/deepzoom/deepzoom_multiserver.py -Q 100 ~/Downloads/CAMELYON16/



