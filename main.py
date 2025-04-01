# Importing required library
from mpl_interactions import ioff, panhandler, zoom_factory
import matplotlib.pyplot as plt

# creating the dataset
data = {'Operating System': 10, 'Data Structure': 7,
		'Machine Learning': 14, 'Deep Learning': 12}
courses = list(data.keys())
values = list(data.values())
# Enable scroll to zoom with the help of MPL
# Interactions library function like ioff and zoom_factory.
with plt.ioff():
	figure, axis = plt.subplots()
# creating the bar plot
plt.xlabel("Courses offered")
plt.ylabel("No. of students enrolled")
plt.title("Students enrolled in different courses")
plt.bar(courses, values, color='green', width=0.4)
disconnect_zoom = zoom_factory(axis)
# Enable scrolling and panning with the help of MPL
# Interactions library function like panhandler.
pan_handler = panhandler(figure)
plt.show()
