import h2tools
import scipy as sp
import numpy as np
import os
import warnings
from heapq import merge
from scipy.linalg import get_lapack_funcs

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
warnings.filterwarnings('ignore')


class additional_matrix_info(object):
    def __init__(self):
        self.pos_row = np.array([])
        self.pos_col = np.array([])
        self.leaves_row = []
        self.leaves_col = []
        self.basis_col = []
        self.basis_row = []
        self.full_basis_r = []
        self.full_basis_c = []
dgesdd = get_lapack_funcs('gesdd', [np.array(1.1)])


class sparse_block(object):
    def __init__(self, matrix, ad_info):
        self.matrix = matrix
        n = matrix.shape[0]
        self.res_bl = []
        self.res_ind = []
        self.ad_info = ad_info
        self.row_list = []  # list of rows, simplifide csr, for marvec
        self.col_list = []  # list of cols, simplifide csc, for matvec
        self.col_shape = []
        self.row_shape = []

    def insert_close(self, show_process=False):
        row_tree = self.matrix.problem.row_tree
        col_tree = self.matrix.problem.col_tree
        self.ad_info.basis_col = [0]*len(col_tree.child)
        self.ad_info.basis_row = [0]*len(row_tree.child)
        self.ad_info.full_basis_r = [0]*len(row_tree.child)
        self.ad_info.full_basis_c = [0]*len(col_tree.child)
        init0 = np.zeros(self.matrix.problem.row_tree.level[-1] + 1, dtype=int)
        self.ad_info.pos_row = init0
        init1 = np.zeros(self.matrix.problem.row_tree.level[-1] + 1, dtype=int)
        self.ad_info.pos_col = init1
        n = self.matrix.shape[0]

        def _add_pos_row(node, pos, pos_tmp, leaves):
            if row_tree.child[node] == []:
                leaves.append(node)
                pos[node] = pos_tmp
                pos_tmp += self.matrix.row_transfer[node].shape[0]
            else:
                pos[node] = pos_tmp
                for i in row_tree.child[node]:
                    pos_tmp = _add_pos_row(i, pos, pos_tmp, leaves)
            return pos_tmp

        def _add_pos_col(node, pos, pos_tmp, leaves):
            if col_tree.child[node] == []:
                leaves.append(node)
                pos[node] = pos_tmp
                pos_tmp += self.matrix.col_transfer[node].shape[0]
            else:
                pos[node] = pos_tmp
                for i in col_tree.child[node]:
                    pos_tmp = _add_pos_col(i, pos, pos_tmp, leaves)
            return pos_tmp
        pos_tmp = 0
        _add_pos_row(0, self.ad_info.pos_row, pos_tmp, self.ad_info.leaves_row)
        self.ad_info.pos_row[-1] = n
        pos_tmp = 0
        _add_pos_col(0, self.ad_info.pos_col, pos_tmp, self.ad_info.leaves_col)
        self.ad_info.pos_col[-1] = n
        row_size = row_tree.level[-1]
        col_size = col_tree.level[-1]
        row_close = self.matrix.problem.row_close
        col_close = self.matrix.problem.col_close
        pos_row = self.ad_info.pos_row
        pos_col = self.ad_info.pos_col
        row_num = [-1] * row_size
        for i in range(len(self.ad_info.leaves_row)):
            row_num[self.ad_info.leaves_row[i]] = i
        col_num = [-1] * col_size
        for i in range(len(self.ad_info.leaves_row)):
            col_num[self.ad_info.leaves_row[i]] = i
        self.row_num = row_num
        self.col_num = col_num
        self.row_list = []
        for i in range(len(self.ad_info.leaves_row)):
            self.row_list.append({})
        self.col_list = []
        for i in range(len(self.ad_info.leaves_col)):
            self.col_list.append({})
        row_ind = 0
        col_ind = 0

        def comp_row_size(i):
            if i in self.ad_info.leaves_row:
                row_size = self.matrix.row_transfer[i].shape[0]
                return row_size
            else:
                row_size = 0
                for p in row_tree.child[i]:
                    row_size += comp_row_size(p)
                return row_size

        for i in range(row_size):
            self.row_shape.append(comp_row_size(i))

        def comp_col_size(i):
            if i in self.ad_info.leaves_col:
                col_size = self.matrix.col_transfer[i].shape[0]
                return col_size
            else:
                col_size = 0
                for p in row_tree.child[i]:
                    col_size += comp_col_size(p)
                return col_size
        for i in range(col_size):
            self.col_shape.append(comp_col_size(i))

        def ins_bl(i, j, bl, block_ind):
            if i in self.ad_info.leaves_row:
                if j in self.ad_info.leaves_col:
                    self.res_bl.append(bl.copy())
                    self.res_ind.append((pos_row[i], pos_col[j]))
                    self.row_list[row_num[i]][col_num[j]] = block_ind
                    self.col_list[col_num[j]][row_num[i]] = block_ind
                    block_ind += 1
                    return block_ind
                else:
                    tmp_pos = 0
                    for p in row_tree.child[j]:
                        block_ind = ins_bl(i, p, bl[:, tmp_pos:tmp_pos +
                                                    self.col_shape[p]],
                                           block_ind)
                        tmp_pos = tmp_pos + self.col_shape[p]
                    return block_ind

            else:
                if j in self.ad_info.leaves_col:
                    tmp_pos = 0
                    for t in row_tree.child[i]:
                        block_ind = ins_bl(t, j, bl[tmp_pos:tmp_pos +
                                                    self.row_shape[t], :],
                                           block_ind)
                        tmp_pos = tmp_pos + self.row_shape[t]
                    return block_ind
                else:
                    print ("Bad case!")
                    tmp_pos1 = 0
                    for t in row_tree.child[i]:
                        tmp_pos2 = 0
                        row_shape = pos_row[t+1] - pos_row[t]
                        for p in row_tree.child[j]:
                            col_shape = pos_col[p+1] - pos_col[p]
                            block_ind = ins_bl(t, p, bl[tmp_pos1:tmp_pos1 +
                                                        row_shape, tmp_pos2:
                                                        tmp_pos2 + col_shape],
                                               block_ind)
                            tmp_pos2 = tmp_pos2 + col_shape
                        tmp_pos1 = tmp_pos1 + row_shape
                    return block_ind
        block_ind = 0
        for i in xrange(row_size):
            if len(row_close[i]) != 0:
                for j in range(len(row_close[i])):
                    mat_ij = self.matrix.row_close[i][j]
                    j1 = row_close[i][j]
                    block_ind = ins_bl(i, j1, mat_ij, block_ind)

    def make_lil(self):
        import scipy as sp
        n = self.matrix.shape[0]
        self.res = sp.sparse.lil_matrix((n, n))
        for i in range(len(self.res_ind)):
            p1, p2 = self.res_ind[i]
            self.res[p1:p1 + self.res_bl[i].shape[0],
                     p2:p2 + self.res_bl[i].shape[1]] = self.res_bl[i]

    def insert_m2l(self, i, show_process=False):
        import scipy as sp
        row_tree = self.matrix.problem.row_tree
        col_tree = self.matrix.problem.col_tree
        level_count = len(col_tree.level)-1
        pos_row = self.ad_info.pos_row
        pos_col = self.ad_info.pos_col
        block_ind = len(self.res_ind)
        for j in range(row_tree.level[level_count-i-2],
                       row_tree.level[level_count-i-1]):
            for l in range(len(self.matrix.problem.col_far[j])):
                j1 = self.matrix.problem.row_far[j][l]

                child_pos1 = 0
                for k in self.ad_info.basis_col[j]:
                    child_pos2 = 0
                    for t in self.ad_info.basis_row[j1]:
                        t_i = self.matrix.col_interaction[j][l]
                        tmp_mat = t_i[child_pos1:child_pos1 + k[1],
                                      child_pos2:child_pos2 + t[1]].T
                        self.res_bl.append(tmp_mat)
                        self.res_ind.append((pos_row[t[0]], pos_col[k[0]]))
                        tmp_row_0 = self.row_num[t[0]]
                        tmp_col_0 = self.col_num[k[0]]
                        self.row_list[tmp_row_0][tmp_col_0] = block_ind
                        self.col_list[tmp_col_0][tmp_row_0] = block_ind
                        block_ind += 1
                        child_pos2 += t[1]
                    child_pos1 += k[1]

    def ldot(self, U, lvl):
        import scipy as sp
        basis = self.ad_info.full_basis_r
        row_tree = self.matrix.problem.row_tree
        level_count = len(row_tree.level)-1
        n = self.matrix.shape[0]
        block_ind = len(self.res_ind)
        for i in range(row_tree.level[level_count - lvl - 2],
                       row_tree.level[level_count - lvl - 1]):
            nonz_bls = []
            u = self.matrix.row_transfer[i]
            if u is not None:
                if u.shape[0] != u.shape[1]:
                    u_old = u
                    u, _, _, _ = dgesdd(u, compute_uv=1, full_matrices=1,
                                        overwrite_a=0)
                    u[:, :u_old.shape[1]] = u_old
                    u_T = u.T
                else:
                    u_T = u.T
                for j in basis[i]:
                    tmp_set = set(self.row_list[self.row_num[j[0]]].keys())
                    nonz_bls = list(set(nonz_bls).union(tmp_set))
                u_sh = []
                for j in basis[i]:
                    u_sh.append(j)
                for k in nonz_bls:
                    # Precompute bl_tmp size:
                    tmp_sh1 = 0
                    for j in basis[i]:
                        if self.row_num[j[0]] in self.col_list[k]:
                            tmp_ind = self.col_list[k][self.row_num[j[0]]]
                            bl_sh1 = (self.res_bl[tmp_ind]).shape[1]
                            if bl_sh1 > tmp_sh1:
                                tmp_sh1 = bl_sh1
                    tmp_sh0 = u_T.shape[1]
                    tmp_bl = np.zeros((tmp_sh0, tmp_sh1))
                    # Fill tmp_bl:
                    j_pos = 0
                    for j in basis[i]:
                        if self.row_num[j[0]] in self.col_list[k]:
                            tmp_row = self.row_num[j[0]]
                            bl = self.res_bl[self.col_list[k][tmp_row]]
                            if bl.shape[0] > j[1]:
                                bl = bl[:j[1], :]
                            tmp_bl[j_pos:j_pos+bl.shape[0], :bl.shape[1]] = bl
                            j_pos += j[1]
                        else:
                            j_pos += j[1]
                    tmp_bl = np.dot(u_T, tmp_bl)
                    j_pos = 0
                    for j in basis[i]:
                        j_row = self.row_num[j[0]]
                        if j_row in self.col_list[k]:
                            bl = self.res_bl[self.col_list[k][j_row]]
                            if bl.shape[0] <= j[1]:
                                if bl.shape[1] <= tmp_bl.shape[1]:
                                    tmp_col = self.col_list[k][j_row]
                                    tmp_bl_0 = tmp_bl[j_pos:j_pos+j[1], :]
                                    self.res_bl[tmp_col] = tmp_bl_0

                                else:
                                    bl = np.vstack((bl,
                                                    np.zeros((j[1] -
                                                              bl.shape[0],
                                                              bl.shape[1]))))
                                    tm_b = tmp_bl.shape[1]
                                    bl[:, :tm_b] = p_bl[j_pos:j_pos + j[1], :]
                                    self.res_bl[self.col_list[k][j_row]] = bl
                            else:
                                if bl.shape[1] >= tmp_bl.shape[1]:
                                    tmp_l = self.col_list[k][j_row]
                                    tm_b = tmp_bl.shape[1]
                                    tm_b0 = tmp_bl[j_pos:j_pos+j[1], :]
                                    self.res_bl[tmp_l][:j[1],
                                                       :tm_b] = tm_b0
                                else:
                                    tm_b = tmp_bl.shape[1]
                                    bl = np.hstack((bl,
                                                    np.zeros((bl.shape[0],
                                                              tm_b -
                                                              bl.shape[1]))))
                                    bl[:j[1], :] = tmp_bl[j_pos:j_pos+j[1], :]
                                    self.res_bl[self.col_list[k][j_row]] = bl
                        else:
                            self.res_bl.append(tmp_bl[j_pos:j_pos+j[1], :])
                            tmp_adin = self.ad_info.leaves_col[k]
                            tm = self.ad_info.pos_col[tmp_adin]
                            self.res_ind.append((self.ad_info.pos_row[j[0]],
                                                 tm))
                            self.row_list[self.row_num[j[0]]][k] = block_ind
                            self.col_list[k][self.row_num[j[0]]] = block_ind
                            block_ind += 1
                        j_pos += j[1]
                U.mat.append((u_T, u_sh))

    def rdot(self, U, lvl):
        import scipy as sp
        basis = self.ad_info.full_basis_c
        col_tree = self.matrix.problem.col_tree
        level_count = len(col_tree.level)-1
        n = self.matrix.shape[0]
        block_ind = len(self.res_ind)
        for i in range(col_tree.level[level_count - lvl - 2],
                       col_tree.level[level_count - lvl - 1]):
            u = self.matrix.col_transfer[i]
            if u is not None:
                if u.shape[0] != u.shape[1]:
                    u_old = u
                    u, _, _, _ = dgesdd(u, compute_uv=1,
                                        full_matrices=1, overwrite_a=0)
                    u[:, :u_old.shape[1]] = u_old
                    u_T = u.T
                else:
                    u_T = u.T
                nonz_bls = []
                for j in basis[i]:
                    tmp_set = set(self.col_list[self.col_num[j[0]]].keys())
                    nonz_bls = list(set(nonz_bls).union(tmp_set))
                u_sh = []
                for j in basis[i]:
                    u_sh.append(j)
                for k in nonz_bls:
                    # Precompute bl_tmp size:
                    tmp_sh0 = 0
                    for j in basis[i]:
                        if self.col_num[j[0]] in self.row_list[k]:
                            tmp_col = self.col_num[j[0]]
                            tmp_list = self.row_list[k][tmp_col]
                            bl_sh0 = (self.res_bl[tmp_list]).shape[0]
                            if bl_sh0 > tmp_sh0:
                                tmp_sh0 = bl_sh0

                    tmp_sh1 = u_T.shape[0]
                    tmp_bl = np.zeros((tmp_sh0, tmp_sh1))
                    # Fill tmp_bl:
                    j_pos = 0
                    for j in basis[i]:
                        if self.col_num[j[0]] in self.row_list[k]:
                            tmp_col = self.row_list[k][self.col_num[j[0]]]
                            bl = self.res_bl[tmp_col]
                            if bl.shape[1] >= j[1]:
                                bl = bl[:, :j[1]]

                            tmp_bl[:bl.shape[0], j_pos:j_pos+bl.shape[1]] = bl
                            j_pos += j[1]
                        else:
                            j_pos += j[1]
                    tmp_bl = np.dot(tmp_bl, u_T.T)
                    j_pos = 0
                    for j in basis[i]:
                        j_col = self.col_num[j[0]]
                        if j_col in self.row_list[k]:
                            bl = self.res_bl[self.row_list[k][j_col]]
                            if bl.shape[1] <= j[1]:
                                if bl.shape[0] <= tmp_bl.shape[0]:
                                    tmp_row = self.row_list[k][j_col]
                                    tb = tmp_bl[:, j_pos:j_pos+j[1]].copy()
                                    self.res_bl[tmp_row] = tb
                                else:
                                    bs = bl.shape
                                    bl = np.hstack((bl, np.zeros(bs[0],
                                                                 j[1] -
                                                                 bs[1])))
                                    tb = tmp_bl[:, j_pos:j_pos+j[1]].copy()
                                    bl[:tmp_bl.shape[0], :] = tb
                                    self.res_bl[self.row_list[k][j_col]] = bl
                            else:
                                if bl.shape[0] >= tmp_bl.shape[0]:
                                    res_pos = self.row_list[k][j_col]
                                    to_ins = tmp_bl[:, j_pos:j_pos+j[1]].copy()
                                    tb0 = tmp_bl.shape[0]
                                    self.res_bl[res_pos][:tb0, :j[1]] = to_ins
                                else:
                                    bl = np.vstack((bl,
                                                    np.zeros((tmp_bl.shape[0] -
                                                              bl.shape[0],
                                                              bl.shape[1]))))
                                    tb = tmp_bl[:, j_pos:j_pos+j[1]].copy()
                                    bl[:, :j[1]] = tb
                                    self.res_bl[self.row_list[k][j_col]] = bl
                        else:
                            self.res_bl.append(tmp_bl[:, j_pos:j_pos+j[1]])
                            tmp_a = self.ad_info.leaves_row[k]
                            res_tmp_bl = self.ad_info.pos_row[tmp_a]
                            self.res_ind.append((res_tmp_bl,
                                                 self.ad_info.pos_col[j[0]]))
                            self.row_list[k][self.col_num[j[0]]] = block_ind
                            self.col_list[self.col_num[j[0]]][k] = block_ind
                            block_ind += 1
                        j_pos += j[1]
                U.mat.append((u_T.T, u_sh))

    def make_csr(self):
        import scipy as sp
        n = self.matrix.shape[0]
        size = 0
        for i in range(len(self.res_ind)):
            size += self.res_bl[i].shape[0] * self.res_bl[i].shape[1]
        nnz = size/n

        data = np.zeros((3, size))
        ind = 0
        for i in range(len(self.res_ind)):
            p1, p2 = self.res_ind[i]
            for j in range(self.res_bl[i].shape[0]):
                for k in range(self.res_bl[i].shape[1]):
                    data[:, ind] = [self.res_bl[i][j, k], p1+j, p2+k]
                    ind += 1
        self.res = sp.sparse.coo_matrix((data[0], (data[1], data[2])),
                                        shape=(n, n))
        self.res = self.res.tocsr()


class block_diag(object):
    import scipy as sp

    def gen_col_new(self, i):
        import scipy as sp
        col_tree = self.matrix.problem.col_tree
        basis_col = self.ad_info.basis_col
        leaves_col = self.ad_info.leaves_col
        level_count = len(col_tree.level)-1
        n = self.matrix.shape[0]
        col_tree = self.matrix.problem.col_tree
        pos_row = self.ad_info.pos_row
        pos_col = self.ad_info.pos_col
        f_b = self.ad_info.full_basis_c
        for j in range(col_tree.level[level_count - i - 2],
                       col_tree.level[level_count - i - 1]):
            if j in leaves_col:
                basis_col[j] = [[j, self.matrix.col_transfer[j].shape[1]]]
                f_b[j] = [[j, self.matrix.col_transfer[j].shape[0]]]
            else:
                if self.matrix.col_transfer[j] is not None:
                    f_b[j] = []
                    for k in col_tree.child[j]:
                        f_b[j] += basis_col[k]
                    basis_size = 0
                    basis_col[j] = []
                    bul = True
                    for t in f_b[j]:
                        basis_size += t[1]
                        if basis_size < self.matrix.col_transfer[j].shape[1]:
                            basis_col[j] += [t]
                        elif bul:
                            tmp_col = self.matrix.col_transfer[j].shape[1]
                            basis_col[j] += [[t[0],
                                              t[1] - (basis_size - tmp_col)]]
                            bul = False
                        else:
                            pass
        return []

    def gen_row_new(self, i):
        import scipy as sp
        row_tree = self.matrix.problem.row_tree
        basis_row = self.ad_info.basis_row
        leaves_row = self.ad_info.leaves_row
        level_count = len(row_tree.level)-1
        n = self.matrix.shape[0]
        row_tree = self.matrix.problem.row_tree
        pos_row = self.ad_info.pos_row
        pos_col = self.ad_info.pos_col
        f_b = self.ad_info.full_basis_r
        for j in range(row_tree.level[level_count - i - 2],
                       row_tree.level[level_count - i - 1]):
            if j in leaves_row:
                basis_row[j] = [[j, self.matrix.row_transfer[j].shape[1]]]
                f_b[j] = [[j, self.matrix.row_transfer[j].shape[0]]]
            else:
                if self.matrix.row_transfer[j] is not None:
                    f_b[j] = []
                    for k in row_tree.child[j]:
                        f_b[j] += basis_row[k]
                    basis_size = 0
                    basis_row[j] = []
                    bul = True
                    for t in f_b[j]:
                        basis_size += t[1]
                        if basis_size < self.matrix.row_transfer[j].shape[1]:
                            basis_row[j] += [t]
                        elif bul:
                            tm_row = self.matrix.row_transfer[j].shape[1]
                            basis_row[j] += [[t[0],
                                              t[1] -
                                              (basis_size - tm_row)]]
                            bul = False
                        else:
                            pass

        return []

    def __init__(self, i, tree_type, matrix, ad_info, param="old"):
        self.matrix = matrix
        self.ad_info = ad_info
        self.mat = []
        if param == "old":
            if tree_type == 'row':
                self.mat = self.gen_row(i)
            if tree_type == 'col':
                self.mat = self.gen_col(i)
        elif param == "new":
            if tree_type == 'row':
                self.mat = self.gen_row_new(i)
            if tree_type == 'col':
                self.mat = self.gen_col_new(i)
        else:
            print ("Wrong param!")

    def make_csr(self):
        import scipy as sp
        n = self.matrix.shape[0]
        size = 0

        for i in self.mat:
            pos_u_r = 0
            for j in i[1]:
                pos_u_c = 0
                for k in i[1]:
                    pos_r = self.ad_info.pos_row[j[0]]
                    pos_c = self.ad_info.pos_col[k[0]]
                    size += j[1] * k[1]
        data = np.zeros((3, size))
        ind = 0
        for i in self.mat:
            mat = i[0] - np.eye(i[0].shape[0])
            pos_u_r = 0
            for j in i[1]:
                pos_u_c = 0
                for k in i[1]:

                    pos_r = self.ad_info.pos_row[j[0]]
                    pos_c = self.ad_info.pos_col[k[0]]
                    u = mat[pos_u_r:pos_u_r + j[1], pos_u_c:pos_u_c + k[1]]
                    for i1 in range(u.shape[0]):
                        for j1 in range(u.shape[1]):
                            data[:, ind] = [u[i1, j1], pos_r + i1, pos_c + j1]
                            ind += 1
                    pos_u_c += k[1]
                pos_u_r += j[1]
        self.mat = sp.sparse.coo_matrix((data[0], (data[1], data[2])),
                                        shape=(n, n)).T
        self.mat = self.mat.tocsr() + sp.sparse.eye(n, format='csr')


def convert_h2_to_sparse(matrix, check_error=False, show_process=False):
    import scipy as sp
    ad_info = additional_matrix_info()
    S = sparse_block(matrix, ad_info)
    U = []
    V = []
    row_tree = matrix.problem.row_tree
    level_count = len(row_tree.level)-1
    S.insert_close(show_process)
    for i in range(0, level_count-1):
        U_tmp = block_diag(i, 'row', matrix, ad_info, param='new')
        S.ldot(U_tmp, i)
        U_tmp.make_csr()
        V_tmp = block_diag(i, 'col', matrix, ad_info, param='new')
        S.rdot(V_tmp, i)
        V_tmp.make_csr()
        S.insert_m2l(i)
        U.append(U_tmp.mat)
        V.append(V_tmp.mat)
    S.make_csr()
    if check_error:
        test_factorization(S.res, U, V, matrix)
    return S, U, V


def inv(perm):
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse


def test_factorization(res, u_mat_row, u_mat_col, matrix):
    n = matrix.shape[0]
    x = np.ones(n)
    y_h2 = matrix.dot(x)
    row_tree = matrix.problem.row_tree
    level_count = len(row_tree.level)-1
    mega_u_col = u_mat_col[0]
    for j in range(1, level_count-1):
        mega_u_col = u_mat_col[j].dot(mega_u_col)
    mega_u_row = u_mat_row[0]
    for j in range(1, level_count-1):
        mega_u_row = mega_u_row.dot(u_mat_row[j])
    prm = row_tree.index.index
    iprm = inv(prm)
    Px = x[prm]
    UPx = mega_u_col.dot(Px)
    RUPx = res.dot(UPx)
    UTRPUx = mega_u_row.dot(RUPx)
    y_sp = UTRPUx[iprm]
    print ('Error:', np.linalg.norm(y_h2 - y_sp))


def jit_init():
    ndim = 2
    count = 100
    tau = 1e-2
    iters = 1
    onfly = 0
    symmetric = 0
    block_size = 10
    verbose = 0
    func = particles.inv_distance
    np.random.seed(0)
    position = np.random.randn(ndim, count)
    data = particles.Particles(ndim, count, position)
    tree1 = ClusterTree(data, block_size)
    problem = Problem(func, tree1, tree1, symmetric, verbose)
    matrix2 = mcbh(problem, tau, iters=iters, onfly=onfly, verbose=verbose)
    matrix2.svdcompress(tau=1e-2)
    S, U, V = convert_h2_to_sparse(matrix2, show_process=False)


def time_S(matrix, show_process=False):
    ad_info = additional_matrix_info()
    S = sparse_block(matrix, ad_info)
    row_tree = matrix.problem.row_tree
    level_count = len(row_tree.level)-1
    S.insert_close(show_process)
    for i in range(0, level_count-1):
        U_tmp = block_diag(i, 'row', matrix, ad_info, param='new')
        S.ldot(U_tmp, i)
        V_tmp = block_diag(i, 'col', matrix, ad_info, param='new')
        S.rdot(V_tmp, i)
        S.insert_m2l(i)
    return 0
