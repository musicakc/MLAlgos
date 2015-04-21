from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot2

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data)
graph = pydot2.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")
