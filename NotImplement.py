# __author__ = 'yuxinsun'
#     def _proba(self, G):
#         for node in G.nodes():
#             s = (np.sum(G[node][x]['kern_unnorm_']) for x in G.successors(node))
#             s = sum(s)
#             for successor_ in G.successors(node):
#                 if s == 0:
#                     G[node][successor_]['proba_'] = 0.
#                 else:
#                     G[node][successor_]['proba_'] = np.sum(G[node][successor_]['kern_unnorm_'])/s
#                 if G[node][successor_]['proba_'] < self.proba_threshold:
#                     G.remove_edge(node, successor_)
#
#         isolated_ = nx.isolates(G)
#         G.remove_nodes_from(isolated_)
#
#         return G
#
#     def _log_proba(self, G):
#         proba_ = nx.get_edge_attributes(G, 'proba_')
#         proba_ = OrderedDict(sorted(proba_.items(), key=lambda t: t[0]))
#         proba_ = 1/np.asarray(proba_.values(), dtype=float)
#         proba_[np.where(proba_ == 0.)] = np.inf
#         log_proba_ = np.log(1/proba_)  # to be checked: isnan?
#
#         self.log_proba_ = log_proba_
#
#     def _normlise_DAG(self, G):
#         kern_ = nx.get_edge_attributes(G, 'kern_unnorm_')
#         kern_ = OrderedDict(sorted(kern_.items(), key=lambda t: t[0]))
#         val_ = np.asarray(kern_.values(), dtype=float)
#         key_ = kern_.keys()
#
#         if len(val_.shape) == 2:
#             kern_ = normalize(val_ * self.log_proba_[:, None], norm='l2', axis=0)
#         else:
#             kern_ = (val_ * self.log_proba_)/np.linalg.norm(kern_)
#
#         kern_ = dict(zip(key_, kern_))
#         nx.set_edge_attributes(G, 'kern_', kern_)
#
#         return G