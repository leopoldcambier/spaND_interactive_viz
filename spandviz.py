import numpy as np
import numpy.linalg as nla
import scipy.linalg as la
import scipy.io
import scipy.sparse
import time

from bokeh.io import output_file, show, curdoc
from bokeh.layouts import column, row
from bokeh.models import GraphRenderer, IndexFilter, ColumnDataSource, CDSView, GlyphRenderer, \
                         Circle, Oval, Plot, StaticLayoutProvider, Slider, TextInput, \
                         RadioButtonGroup, Button, Select, CheckboxGroup, Paragraph, Div
from bokeh.plotting import figure
from bokeh.palettes import viridis

class TrailingMatrix:

    def complement(self, dofs):
        other_dofs = np.array([i for i in range(self.A.shape[0]) if i not in dofs])
        Asn = self.submatrix(dofs, other_dofs)
        Ans = self.submatrix(other_dofs, dofs)
        Asn_col_norms = nla.norm(Asn, axis=0, ord=np.inf)
        Asn_other_dofs = other_dofs[Asn_col_norms > 0]
        AnsT_col_norms = nla.norm(Ans.T, axis=0, ord=np.inf)
        AnsT_other_dofs = other_dofs[AnsT_col_norms > 0]
        other_dofs = np.unique(np.concatenate([Asn_other_dofs, AnsT_other_dofs]))
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
        return Asn_ns

    def get_svds(self, dofs):
        Asn_ns = self.edges(dofs)
        (U, s, VT) = la.svd(Asn_ns)
        return s
    
    # Sparsify
    def sparsify(self, dofs, tol=0.5):
        
        other_dofs = self.complement(dofs)
        self.scale(dofs)

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

        other_dofs = self.complement(dofs)
        self.scale(dofs)
        
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
                      tools = "box_select",
                      output_backend="webgl",
                      width=500,
                      height=500,
                      )

    def get_figure_trailing_matrix(self):
        return figure(title='Matrix',
                      x_range=(0, self.N),
                      y_range=(0, self.N),
                      output_backend="webgl",
                      width=500,
                      height=500,
                      )

    def get_figure_svds_plot(self):
        return figure(title='Singular values of edges of selected vertices',
                      y_axis_type="log",
                      output_backend="webgl",
                      width=500,
                      height=500,
                      )

    def __init__(self, A_file, X_file, stepping, p_file):

        # Load
        A = scipy.io.mmread(A_file).todense()
        X = scipy.io.mmread(X_file)
        self.N = A.shape[0]
        self.stepping = stepping
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
        self.input_dofs = TextInput(title="Vertices to eliminate/sparsify (comma separated list)", value='', width=250)
        self.input_dofs.on_change('value', self.callback_highlight)
        reset_button = Button(label="Reset selection", button_type="success", width=250)
        reset_button.on_click(self.callback_clear_selection)
        eliminate_button = Button(label="Eliminate", button_type="success", width=250)
        eliminate_button.on_click(self.callback_eliminate)
        sparsify_button  = Button(label="Sparsify", button_type="success", width=250)
        sparsify_button.on_click(self.callback_sparsify)
        step_button = Button(label="Step and eliminate", button_type="success", width=250)
        step_button.on_click(self.callback_step)
        self.options_checkbox = CheckboxGroup(labels=["Show edges","Show singular values"], active=[0, 1], width=250)
        self.options_checkbox.on_change('active', self.callback_checkbox)
        self.step_size = TextInput(value="10", title="Step size")

        # Assemble all
        if stepping == "manual":
            buttons     = column(self.input_dofs, reset_button, row(eliminate_button, sparsify_button, self.options_checkbox))
            data        = row(self.get_figure_graph_matrix(), self.get_figure_trailing_matrix(), self.get_figure_svds_plot())
            self.layout = column(buttons, data)
        else:
            buttons     = column(row(step_button, self.options_checkbox), self.step_size)
            data        = row(self.get_figure_graph_matrix(), self.get_figure_trailing_matrix())
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

    def callback_clear_selection(self, event):
        self.clear_selection()

    def clear_selection(self):
        self.input_dofs.value = ""

    def callback_checkbox(self, attr, old, new):
        self.update_plot()

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
                alpha=[1.0 for i in active],
            ))
        else:
            nodes_source = ColumnDataSource(dict(
                index=active,
                x=self.X[0,active],
                y=self.X[1,active],
                color=['black' for i in active],
                alpha=[1.0 if i in highlighted else 0.1 for i in active],
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
            alpha="alpha",
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


        if self.stepping == "manual":
            
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
        else:
            self.layout.children[1].children = [graph_matrix, trailing_matrix]

#### Top level app

select = Select(value="Poisson 5x5", options=["Poisson 5x5", "Poisson 16x16", "Poisson 32x32", "Naca 8"], width=200)
ordering = RadioButtonGroup(labels=["Manual ordering", "Nested Dissection", "Topological"], active=0, width=400)
title = Div(text="""<h1>spaND visualizationt tool</h1><p>Scroll down for help</p>""")

description = Div(text=\
    """ <p>This tools lets you explore elimination (and sparsification) orders and how they affect a matrix sparsity pattern.
        The central figures show
        <ul><li>Left: the matrix graph (i.e., there is an edge between i and j if A(i,j) or A(j,i) are non zero in the (trailing) matrix)</li>
            <li>Center: the matrix itself, with zero entries in purple and non zero entries in yellow</li>
            <li>Right: if active, it shows the singular values of the [A(s,n) A(n,s)^T] block, where s are the select vertices and n are all their neighbors in the matrix graph</li>
        </ul>
        How to use it:        
        <ol>
        <li>Select a problem at the very top by choosing between the three proposed problems</li>
        <li>You can then pick between manually chosing nodes to eliminated ("Manual ordering") or using a predefined\
            "Nested Dissection" (good) or "Topological" (arbitrary, bad in general) ordering</li>
        <li><ul>
            <li>In the manual case: select nodes by using the "box selection" tool, then click on "Eliminate". You can reset your selection with the "Reset selection" button.</li>
            <li>In the predefined ordering case, simply select how many nodes to eliminated using the box and click on "Step and eliminate" to eliminate the next 'x' vertices
            </ul></li>
        <li>To make plotting faster you can disable edges plotting (uncheck "Show edges") and singular values display ("Show singular values")</li></ol></p>""", width=400, height=100)

def update():
    ordering_kind = ordering.active
    
    if ordering_kind == 0:
        stepping_kind = "manual"
    else:
        stepping_kind = "step"

    if(select.value == "Poisson 5x5"):
        matrix_file = "matrices/neglapl_2_5.mm"
        coo_file = "matrices/5x5.mm"
    elif(select.value == "Poisson 16x16"):
        matrix_file = "matrices/neglapl_2_16.mm"
        coo_file = "matrices/16x16.mm"
    elif(select.value == "Poisson 32x32"):
        matrix_file = "matrices/neglapl_2_32.mm"
        coo_file = "matrices/32x32.mm"
    elif(select.value == "Naca 8"):
        matrix_file = "matrices/naca8_jac_trimmed.mm"
        coo_file = "matrices/naca8_coo_trimmed.mm"
    
    if ordering_kind == 1: # ND
        perm_file = matrix_file + ".ndperm"
    else:
        perm_file = None
    
    sv = SpandVisualizer(matrix_file, coo_file, stepping_kind, perm_file)
    document = column(title, row(select, ordering), sv.get_layout(), description)
    curdoc().clear()
    curdoc().add_root(document)
    curdoc().title = "spaND viz"

def callback_menu(attr, old, new):
    print("Menu callback")
    update()

select.on_change('value', callback_menu)
ordering.on_change('active', callback_menu)

update()