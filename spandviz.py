import numpy as np
import scipy.linalg as la
import scipy.io
import scipy.sparse

from bokeh.io import output_file, show, curdoc
from bokeh.layouts import column, row
from bokeh.models import GraphRenderer, IndexFilter, ColumnDataSource, CDSView, GlyphRenderer, Circle, Oval, Plot, StaticLayoutProvider, Slider, TextInput, RadioButtonGroup, Button, Select
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

    def __init__(self, A_file, X_file):

        # Load
        A = scipy.io.mmread(A_file).todense()
        X = scipy.io.mmread(X_file)
        self.N = A.shape[0]
        # The trailing matrix in the natural order
        self.A = TrailingMatrix(A)
        # The coordinates in the natural order
        self.X = X
        # The dofs not-yet eliminated in natural order
        all_dofs = list(range(self.N))
        self.active_dofs = all_dofs

        # Initialize plot
        self.graph_matrix = figure(title='Graph', 
                                   x_range=(np.amin(X[0,:])-1,np.amax(X[0,:])+1), 
                                   y_range=(np.amin(X[1,:])-1,np.amax(X[1,:])+1), 
                                   tooltips=[("index", "$index")],
                                   tools = "pan,wheel_zoom,box_zoom,reset,box_select,lasso_select",
                                   output_backend="webgl"
                                  )

        self.trailing_matrix = figure(title='Matrix',
                                      x_range=(0, self.N),
                                      y_range=(0, self.N),
                                      output_backend="webgl"
                                     )

        # Set data sources
        self.nodes_source = ColumnDataSource(dict(
            index=all_dofs,
            x=self.X[0,:],
            y=self.X[1,:],
        ))
        self.nodes_source.selected.on_change('indices', self.callback_select)

        edges = scipy.sparse.coo_matrix(self.A.A)
        edges_source = ColumnDataSource(dict(start=edges.row,end=edges.col))

        layout = StaticLayoutProvider(graph_layout=dict(zip(all_dofs, zip(self.X[0,:],self.X[1,:]))))

        nodes_view = CDSView(source=self.nodes_source, filters=[IndexFilter([])])

        # highlight_source = ColumnDataSource(dict(
        #     index=[],
        #     x=[],
        #     y=[],
        #     color=[]
        # ))

        # Buttons and widgets
        self.input_dofs = TextInput(title="Dofs to eliminate/sparsify (comma separated list)", value='')
        self.input_dofs.on_change('value', self.callback_highlight)
        self.eliminate_button = Button(label="Eliminate", button_type="success")
        self.eliminate_button.on_click(self.callback_eliminate)
        self.sparsify_button  = Button(label="Sparsify", button_type="success")
        self.sparsify_button.on_click(self.callback_sparsify)

        # Left graph
        self.graph_matrix.graph(
            self.nodes_source,
            edges_source,
            layout,
        )

        self.graph_matrix.scatter(
            x="x",
            y="y",
            color="black",
            source=self.nodes_source,
            view=nodes_view,
        )

        # Right picture
        self.trailing_matrix.image(
            image=[self.get_mat()], 
            x=0, 
            y=0, 
            dh=self.N, 
            dw=self.N, 
            palette=viridis(256)
        )

        # Assemble all
        self.row_dofs   = row(self.input_dofs, width=600)
        self.row_choice = row(self.eliminate_button, self.sparsify_button, width=600)
        self.row_data   = row(self.graph_matrix, self.trailing_matrix, width=1400)

        # Refresh
        self.update_plot()

    def eliminate(self, dofs):
        self.A.eliminate(dofs)
        self.active_dofs = [d for d in self.active_dofs if (d not in dofs)]

    def sparsify(self, dofs, tol=0.5):
        rank = self.A.sparsify(dofs, tol)
        self.active_dofs = [d for d in self.active_dofs if (d not in dofs[rank:])]

    def callback_eliminate(self, event):
        print("Eliminate callback")
        dofs = [int(i) for i in self.input_dofs.value.split(',')]
        self.eliminate(dofs)
        self.clear_selection()
        self.update_plot()

    def callback_sparsify(self, event):
        print("Sparsify callback")
        dofs = [int(i) for i in self.input_dofs.value.split(',')]
        self.sparsify(dofs, 0.5)
        self.clear_selection()
        self.update_plot()

    def clear_selection(self):
        self.input_dofs.value = ""

    def callback_select(self, attr, old, new):
        print("Select callback")
        selected = [self.nodes_source.data['index'][i] for i in new]
        active_selected = [str(i) for i in sorted(list(set(self.active_dofs) & set(selected)))]
        self.input_dofs.value = ",".join(active_selected)

    def callback_highlight(self, attr, old, new):
        print("Highlight callback")
        self.update_plot()

    def highlighted_dofs(self):
        if(len(self.input_dofs.value)) > 0:
            highlight = [int(i) for i in self.input_dofs.value.split(',')]
        else:
            highlight = []
        return highlight

    # Return right image
    def get_mat(self):
        highlight = self.highlighted_dofs()
        matA = self.A.A
        matA = (matA != 0).astype(float)
        matA[np.ix_(highlight, highlight)] = 0.5 * matA[np.ix_(highlight, highlight)]
        mat = np.identity(self.N)
        n_inactive = self.N - len(self.active_dofs)
        mat[n_inactive:,n_inactive:] = matA[np.ix_(self.active_dofs, self.active_dofs)]
        mat = mat[::-1,:]
        return mat

    def get_edges(self):
        Asp = scipy.sparse.coo_matrix(self.A.A)
        return Asp.tocoo()

    def update_plot(self):

        # Update edges
        edges = self.get_edges()
        row = [i for i in edges.row if i in self.active_dofs]
        col = [i for i in edges.col if i in self.active_dofs]
        self.graph_matrix.renderers[0].edge_renderer.data_source.data = dict(
            start=row,
            end=col
        )

        # Dofs to highlight
        highlight = self.highlighted_dofs()
        self.graph_matrix.renderers[1].view = CDSView(source=self.nodes_source, filters=[IndexFilter(highlight)])

        # Update image
        self.trailing_matrix.image(
            image=[self.get_mat()], 
            x=0, 
            y=0, 
            dh=self.N, 
            dw=self.N, 
            palette=viridis(256)
        )

#### Top level app

select = Select(value="Poisson 5x5", options=["Poisson 5x5", "Poisson 16x16", "Naca 8"])

def update():
    if(select.value == "Poisson 5x5"):
        sv = SpandVisualizer("neglapl_2_5.mm", "5x5.mm")
    elif(select.value == "Poisson 16x16"):
        sv = SpandVisualizer("neglapl_2_16.mm", "16x16.mm")
    elif(select.value == "Naca 8"):
        sv = SpandVisualizer("naca8_jac_trimmed.mm", "naca8_coo_trimmed.mm")
    document = column(select, sv.row_dofs, sv.row_choice, sv.row_data)
    curdoc().clear()
    curdoc().add_root(document)
    curdoc().title = "spaND viz"

def callback_menu(attr, old, new):
    print("Menu callback")
    update()

select.on_change('value', callback_menu)

update()