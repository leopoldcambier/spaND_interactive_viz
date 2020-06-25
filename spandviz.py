import numpy as np
import scipy.linalg as la
import scipy.io
import scipy.sparse

from bokeh.io import output_file, show, curdoc
from bokeh.layouts import column, row
from bokeh.models import GraphRenderer, Oval, StaticLayoutProvider, Slider, TextInput, RadioButtonGroup, Button
from bokeh.models.tools import LassoSelectTool, BoxSelectTool
from bokeh.plotting import figure
from bokeh.palettes import viridis

class TrailingMatrix:

    def complement(self, dofs):
        other_dofs = [i for i in range(self.A.shape[0]) if i not in dofs]
        return other_dofs
    
    def submatrix(self, n1, n2):
        return self.A[np.ix_(n1, n2)]

    def __init__(self, A):
      self.A = A

    def get_rank(self, s, tol):
        if(len(s) == 0):
            return 0
        if(tol > 1.0):
            return 0
        s = s / s[0]
        rank = np.sum(s >= tol)
        return rank

    # Scale block using PLU
    def scale(self, dofs):

        other_dofs = self.complement(dofs)
        Ass = self.submatrix(dofs, dofs)
        Asn = self.submatrix(dofs, other_dofs)
        Ans = self.submatrix(other_dofs, dofs)

        (P, L, U) = la.lu(Ass)
        LnsT = la.solve_triangular(U.T, Ans.T, lower=True) # Lns = Ans U^-1 <-> Lns U = Ans <-> U^T Lns^T = Ans^T
        Lns  = LnsT.T
        PAsn = np.matmul(P.T, Asn)
        Lsn  = la.solve_triangular(L, PAsn, lower=True, unit_diagonal=True) # Lsn = L^-1 P^-1 Asn <-> L Lsn = P^-1 Asn

        self.A[np.ix_(dofs, other_dofs)] = Lsn
        self.A[np.ix_(other_dofs, dofs)] = Lns
        self.A[np.ix_(dofs, dofs)]       = np.identity(len(dofs))

    # Sparsify
    def sparsify(self, dofs, tol=0.5):
        
        self.scale(dofs)
        other_dofs = self.complement(dofs)

        Ass = self.submatrix(dofs, dofs)
        Asn = self.submatrix(dofs, other_dofs)
        Ans = self.submatrix(other_dofs, dofs)

        Asn_ns = np.concatenate((Asn, Ans.T), axis=1)

        (U, s, VT) = la.svd(Asn_ns)
        r = self.get_rank(s, tol)
        Wsn_ns = np.matmul(np.diag(s[:r]), VT[:r,:])

        n = len(other_dofs)
        Wsn = Wsn_ns[:,:n]
        Wns = Wsn_ns[:,n:].T

        self.A[np.ix_(dofs[:r], other_dofs)] = Wsn
        self.A[np.ix_(dofs[r:], other_dofs)] = 0.0
        self.A[np.ix_(other_dofs, dofs[:r])] = Wns
        self.A[np.ix_(other_dofs, dofs[r:])] = 0.0

        return r

    # Eliminate
    def eliminate(self, dofs):

        self.scale(dofs)
        other_dofs = self.complement(dofs)
        
        Asn = self.submatrix(dofs, other_dofs)
        Ans = self.submatrix(other_dofs, dofs)
        Ann = self.submatrix(other_dofs, other_dofs)

        self.A[np.ix_(other_dofs, other_dofs)] = Ann - np.matmul(Ans, Asn)
        self.A[np.ix_(other_dofs, dofs)]       = 0.0
        self.A[np.ix_(dofs, other_dofs)]       = 0.0

class SpandVisualizer:

    def __init__(self, A, X):
        self.N = A.shape[0]
        # the trailing matrix in the natural order
        self.A = TrailingMatrix(A)
        # the coordinates in the natural order
        self.X = X
        # the dofs not-yet eliminated in natural order
        self.active_dofs = list(range(self.N))
        # map dofs in TrailingMatrix to coordinate
        self.dofs_to_viz_dofs  = np.arange(self.N)

        # Initialize plot
        self.graph_matrix = figure(title='Graph', 
                           x_range=(np.amin(X[0,:])-1,np.amax(X[0,:])+1), 
                           y_range=(np.amin(X[1,:])-1,np.amax(X[1,:])+1), 
                           tooltips=[("index", "$index"),("(x,y)", "($x, $y)")],
                        #    tools=[BoxSelectTool(), LassoSelectTool()],
                           tools = "pan,wheel_zoom,box_zoom,reset,box_select,lasso_select"
                           )

        self.trailing_matrix = figure(title='Matrix',
                                      x_range=(0, self.N),
                                      y_range=(0, self.N)
                                      )

        self.input_dofs = TextInput(title="Dofs to eliminate/sparsify (comma separated list)", value='')
        self.input_dofs.on_change('value', self.callback_highlight)
        self.eliminate_button = Button(label="Eliminate", button_type="success")
        self.eliminate_button.on_click(self.callback_eliminate)
        self.sparsify_button  = Button(label="Sparsify", button_type="success")
        self.sparsify_button.on_click(self.callback_sparsify)

        self.update_plot()
        dofs   = row(self.input_dofs, width=600)
        choice = row(self.eliminate_button, self.sparsify_button, width=600)
        data   = row(self.graph_matrix, self.trailing_matrix, width=1400)
        curdoc().add_root(column(dofs, choice, data))
        curdoc().title = "spaND viz"

    def callback_eliminate(self, event):
        print("Eliminate callback")
        dofs = [int(i) for i in self.input_dofs.value.split(',')]
        self.eliminate(dofs)
        self.update_plot()

    def callback_sparsify(self, event):
        print("Sparsify callback")
        dofs = [int(i) for i in self.input_dofs.value.split(',')]
        self.sparsify(dofs, 0.5)
        self.update_plot()

    def callback_select(self, attr, old, new):
        print("Select callback")
        dofs = [str(self.graph.node_renderer.data_source.data['index'][i]) for i in new]
        self.input_dofs.value = ",".join(dofs)
        self.update_plot()

    def callback_highlight(self, attr, old, new):
        self.update_plot()

    def highlighted_dofs(self):
        if(len(self.input_dofs.value)) > 0:
            highlight = [int(i) for i in self.input_dofs.value.split(',')]
        else:
            highlight = []
        return highlight

    def update_plot(self):

        # Dofs to highlight
        highlight = self.highlighted_dofs()

        # Update graph
        dofs  = self.active_dofs
        edges = scipy.sparse.coo_matrix(self.A.A)

        colors = ['black' for i in dofs]
        for (i,d) in enumerate(dofs):
            if(d in highlight):
                colors[i] = 'red'

        self.graph = GraphRenderer()
        self.graph.node_renderer.data_source.data = dict(index=dofs,x=X[0,dofs],y=X[1,dofs],colors=colors)
        self.graph.node_renderer.glyph.fill_color = "colors"
        self.graph.node_renderer.glyph.line_color = "colors"
        self.graph.node_renderer.data_source.selected.on_change('indices', self.callback_select)
        self.graph.edge_renderer.data_source.data = dict(start=edges.row,end=edges.col)
        graph_layout = dict(zip(dofs, zip(X[0,dofs],X[1,dofs])))
        self.graph.layout_provider = StaticLayoutProvider(graph_layout=graph_layout)
        self.graph_matrix.renderers = [self.graph]

        # Update image
        matA = self.A.A
        matA = (matA != 0).astype(float)
        matA[np.ix_(highlight, highlight)] = 0.5 * matA[np.ix_(highlight, highlight)]

        mat = np.identity(self.N)
        n_inactive = self.N - len(self.active_dofs)
        mat[n_inactive:,n_inactive:] = matA[np.ix_(self.active_dofs, self.active_dofs)]
        mat = mat[::-1,:]
        self.trailing_matrix.image(image=[mat], x=0, y=0, dh=self.N, dw=self.N, palette=viridis(256))

    def eliminate(self, dofs):
        self.A.eliminate(dofs)
        self.active_dofs = [d for d in self.active_dofs if (d not in dofs)]

    def sparsify(self, dofs, tol=0.5):
        rank = self.A.sparsify(dofs, tol)
        self.active_dofs = [d for d in self.active_dofs if (d not in dofs[rank:])]
        # Later we will improve this based on coordinates


A = scipy.io.mmread("neglapl_2_5.mm").todense()
X = scipy.io.mmread("5x5mm")
sv = SpandVisualizer(A, X)