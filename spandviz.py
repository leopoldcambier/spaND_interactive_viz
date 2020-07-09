import numpy as np
import numpy.linalg as nla
import scipy.linalg as la
import scipy.io
import scipy.sparse

from bokeh.io import output_file, show, curdoc
from bokeh.layouts import column, row
from bokeh.models import GraphRenderer, IndexFilter, ColumnDataSource, CDSView, GlyphRenderer, \
                         Circle, Oval, Plot, StaticLayoutProvider, Slider, TextInput, \
                         RadioButtonGroup, Button, Select, CheckboxGroup
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

    # Get edges
    def edges(self, dofs):
        other_dofs = self.complement(dofs)
        Asn = self.submatrix(dofs, other_dofs)
        Ans = self.submatrix(other_dofs, dofs)
        Asn_ns = np.concatenate((Asn, Ans.T), axis=1)
        col_norms = nla.norm(Asn_ns, axis=0, ord=np.inf)
        maxi = np.amax(col_norms)
        Asn_ns_trimmed = Asn_ns[:,col_norms > 1e-12 * maxi]
        return Asn_ns_trimmed

    def get_svds(self, dofs):
        Asn_ns = self.edges(dofs)
        (U, s, VT) = la.svd(Asn_ns)
        return s
    
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

    def get_figure_graph_matrix(self):
        return figure(title='Graph', 
                      x_range=(np.amin(self.X[0,:])-1,np.amax(self.X[0,:])+1), 
                      y_range=(np.amin(self.X[1,:])-1,np.amax(self.X[1,:])+1),
                      tools = "pan,wheel_zoom,box_zoom,reset,box_select,lasso_select",
                      output_backend="webgl"
                      )

    def get_figure_trailing_matrix(self):
        return figure(title='Matrix',
                      x_range=(0, self.N),
                      y_range=(0, self.N),
                      output_backend="webgl"
                      )

    def get_figure_svds_plot(self):
        return figure(title='Singular values of edges of selected vertices',
                      y_axis_type="log",
                      output_backend="webgl"
                      )

    def __init__(self, A_file, X_file, stepping, p_file):

        # Load
        A = scipy.io.mmread(A_file).todense()
        X = scipy.io.mmread(X_file)
        self.N = A.shape[0]
        if(p_file is not None):
            perm = scipy.io.mmread(p_file).reshape((-1,))
        else:
            perm = np.arange(self.N)
        A = A[np.ix_(perm,perm)]
        X = X[:,perm]
        # The trailing matrix in the natural order
        self.A = TrailingMatrix(A)
        # The coordinates in the natural order
        self.X = X
        # The dofs not-yet eliminated in natural order
        all_dofs = list(range(self.N))
        self.active_dofs = all_dofs

        # Buttons and widgets
        self.input_dofs = TextInput(title="Dofs to eliminate/sparsify (comma separated list)", value='')
        self.input_dofs.on_change('value', self.callback_highlight)
        eliminate_button = Button(label="Eliminate", button_type="success")
        eliminate_button.on_click(self.callback_eliminate)
        sparsify_button  = Button(label="Sparsify", button_type="success")
        sparsify_button.on_click(self.callback_sparsify)
        step_button = Button(label="Step and eliminate", button_type="success")
        step_button.on_click(self.callback_step)
        self.options_checkbox = CheckboxGroup(labels=["Show edges","Show singular values"], active=[1])
        self.step_size = TextInput(value="10", title="Step size")

        # Assemble all
        if stepping == "manual":
            buttons     = column(row(self.input_dofs, width=600), row(eliminate_button, sparsify_button, self.options_checkbox, width=900))
            data        = row(self.get_figure_graph_matrix(), self.get_figure_trailing_matrix(), self.get_figure_svds_plot(), width=1800)
            self.layout = column(buttons, data)
        else:
            buttons     = column(row(step_button, self.options_checkbox, width=600), row(self.step_size, width=300))
            data        = row(self.get_figure_graph_matrix(), self.get_figure_trailing_matrix(), self.get_figure_svds_plot(), width=1800)
            self.layout = column(buttons, data)


        # Refresh
        self.update_plot()

    def get_layout(self):
        return self.layout

    def callback_step(self, event):
        size = int(self.step_size.value)
        print("Stepping by ", size)
        dofs = self.active_dofs[:min(len(self.active_dofs),size)]
        self.eliminate(dofs)
        self.update_plot()

    def eliminate(self, dofs):
        print("Eliminating ", dofs)
        self.A.eliminate(dofs)
        self.active_dofs = [d for d in self.active_dofs if (d not in dofs)]

    def sparsify(self, dofs, tol=0.5):
        print("Sparsifying ", dofs)
        rank = self.A.sparsify(dofs, tol)
        self.active_dofs = [d for d in self.active_dofs if (d not in dofs[rank:])]

    def callback_eliminate(self, event):
        print("Eliminate callback")
        dofs = [int(i) for i in self.input_dofs.value.split(',')]
        self.eliminate(dofs)
        self.clear_selection()

    def callback_sparsify(self, event):
        print("Sparsify callback")
        dofs = [int(i) for i in self.input_dofs.value.split(',')]
        self.sparsify(dofs, 0.5)
        self.clear_selection()

    def clear_selection(self):
        self.input_dofs.value = ""

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

        print("Updating")
        active = self.active_dofs
        highlighted = self.highlighted_dofs()

        # Update nodes
        if len(highlighted) == 0:
            nodes_source = ColumnDataSource(dict(
                index=active,
                x=self.X[0,active],
                y=self.X[1,active],
                color=['black' for i in active],
            ))
        else:
            nodes_source = ColumnDataSource(dict(
                index=highlighted,
                x=self.X[0,highlighted],
                y=self.X[1,highlighted],
                color=['black' for i in highlighted],
            ))

        def callback_select(attr, old, new):
            print("Select callback")
            selected = [nodes_source.data['index'][i] for i in new]
            active_selected = list(sorted(list(set(active) & set(selected))))
            active_selected_str = [str(i) for i in active_selected]
            self.input_dofs.value = ",".join(active_selected_str)
            self.update_plot()

        nodes_source.selected.on_change('indices', callback_select)

        # Update edges
        if(0 in self.options_checkbox.active):
            edges = self.get_edges()
            row = [i for i in edges.row if i in active]
            col = [i for i in edges.col if i in active]
            edges_source = ColumnDataSource(dict(
                xs=[ [self.X[0,i],self.X[0,j]] for (i,j) in zip(row,col) ],
                ys=[ [self.X[1,i],self.X[1,j]] for (i,j) in zip(row,col) ]
            ))
        else:
            edges_source = ColumnDataSource(dict(
                xs=[],
                ys=[],
            ))
    

        # Left graph
        graph_matrix = self.get_figure_graph_matrix()
        graph_matrix.multi_line(
            xs="xs",
            ys="ys",
            source=edges_source
        )
        graph_matrix.scatter(
            x="x",
            y="y",
            color="color",
            source=nodes_source,
        )

        # Middle picture
        trailing_matrix = self.get_figure_trailing_matrix()
        trailing_matrix.image(
            image=[self.get_mat()], 
            x=0, 
            y=0, 
            dh=self.N, 
            dw=self.N, 
            palette=viridis(256)
        )

        # Svds
        svds = []
        if(1 in self.options_checkbox.active):
            active_selected_int = self.highlighted_dofs()
            if(len(active_selected_int) > 0):
                svds = self.A.get_svds(active_selected_int)
                if(len(svds) > 0):
                    svds = svds / svds[0]
        svds_source = ColumnDataSource(dict(
            x=np.arange(len(svds)),
            y=svds,
        ))

        svds_plot = self.get_figure_svds_plot()
        svds_plot.line(
            x="x",
            y="y",
            color="black",
            source=svds_source,
        )

        self.layout.children[1].children = [graph_matrix, trailing_matrix, svds_plot]

#### Top level app

select = Select(value="Poisson 5x5", options=["Poisson 5x5", "Poisson 16x16", "Poisson 32x32", "Naca 8"])
ordering = RadioButtonGroup(labels=["Manual ordering", "Nested Dissection", "Topological"], active=0)

def update():
    ordering_kind = ordering.active
    
    if ordering_kind == 0:
        stepping_kind = "manual"
    else:
        stepping_kind = "step"

    if(select.value == "Poisson 5x5"):
        if ordering_kind == 1: # ND
            sv = SpandVisualizer("neglapl_2_5.mm", "5x5.mm", stepping_kind, "neglapl_2_5.mm.ndperm")
        else:
            sv = SpandVisualizer("neglapl_2_5.mm", "5x5.mm", stepping_kind, None)
    elif(select.value == "Poisson 16x16"):
        if ordering_kind == 1: # ND
            sv = SpandVisualizer("neglapl_2_16.mm", "16x16.mm", stepping_kind, "neglapl_2_16.mm.ndperm")
        else:
            sv = SpandVisualizer("neglapl_2_16.mm", "16x16.mm", stepping_kind, None)
    elif(select.value == "Poisson 32x32"):
        if ordering_kind == 1: # ND
            sv = SpandVisualizer("neglapl_2_32.mm", "32x32.mm", stepping_kind, "neglapl_2_32.mm.ndperm")
        else:
            sv = SpandVisualizer("neglapl_2_32.mm", "32x32.mm", stepping_kind, None)
    elif(select.value == "Naca 8"):
        if ordering_kind == 1: # ND
            sv = SpandVisualizer("naca8_jac_trimmed.mm", "naca8_coo_trimmed.mm", stepping_kind, "naca8_jac_trimmed.mm.ndperm")
        else:
            sv = SpandVisualizer("naca8_jac_trimmed.mm", "naca8_coo_trimmed.mm", stepping_kind, None)
    document = column(row(select, width=600), row(ordering, width=600), sv.get_layout())
    curdoc().clear()
    curdoc().add_root(document)
    curdoc().title = "spaND viz"

def callback_menu(attr, old, new):
    print("Menu callback")
    update()

select.on_change('value', callback_menu)
ordering.on_change('active', callback_menu)

update()