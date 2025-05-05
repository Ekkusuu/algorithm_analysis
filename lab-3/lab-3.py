import tkinter as tk
from tkinter import ttk
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import numpy as np
import random
import heapq

class GraphExplorerUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Graph Explorer")
        self.geometry("1100x650")
        self.focus_set()

        # Animation state
        self.steps = []
        self.step_index = 0
        self.current_path = []
        self.current_edges = []
        self.animating = False
        self._holding = False
        self._auto_after_id = None

        # Floyd–Warshall data
        self.fw_next = None
        self.fw_done = False

        # ─── Control Panel ────────────────────────────────────────────────────
        ctrl = ttk.Frame(self, padding=10)
        ctrl.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(ctrl, text="Graph type:").pack(anchor="w")
        self.graph_type = tk.StringVar(value="Tree")
        types = ["Tree", "Sparse", "Dense", "Complete", "Grid"]
        ttk.OptionMenu(ctrl, self.graph_type, types[0], *types).pack(fill="x", pady=(0,10))

        self.is_directed = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="Directed edges", variable=self.is_directed)\
            .pack(anchor="w", pady=(0,10))

        self.is_weighted = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="Weighted edges", variable=self.is_weighted)\
            .pack(anchor="w", pady=(0,20))

        ttk.Label(ctrl, text="Number of nodes:").pack(anchor="w")
        self.node_scale = tk.Scale(ctrl, from_=2, to=200, orient=tk.HORIZONTAL,
                                   command=self._on_nodes_change)
        self.node_scale.set(20)
        self.node_scale.pack(fill="x", pady=(0,10))

        ttk.Label(ctrl, text="Start node:").pack(anchor="w")
        self.start_scale = tk.Scale(ctrl, from_=0, to=self.node_scale.get()-1,
                                    orient=tk.HORIZONTAL, command=self._on_slider_change)
        self.start_scale.set(0)
        self.start_scale.pack(fill="x", pady=(0,10))

        ttk.Label(ctrl, text="End node:").pack(anchor="w")
        self.end_scale = tk.Scale(ctrl, from_=0, to=self.node_scale.get()-1,
                                  orient=tk.HORIZONTAL, command=self._on_slider_change)
        self.end_scale.set(self.node_scale.get()-1)
        self.end_scale.pack(fill="x", pady=(0,20))

        ttk.Button(ctrl, text="Generate Graph", command=self._on_generate)\
            .pack(fill="x", pady=(0,10))

        ttk.Button(ctrl, text="BFS vs DFS", command=lambda: self._interrupt_and(self._on_plot_bfs_dfs))\
            .pack(fill="x", pady=(0,5))
        ttk.Button(ctrl, text="Dijkstra vs Floyd–Warshall", 
                   command=lambda: self._interrupt_and(self._on_plot_sp))\
            .pack(fill="x", pady=(0,5))
        ttk.Button(ctrl, text="Prim vs Kruskal", command=lambda: self._interrupt_and(self._on_plot_mst))\
            .pack(fill="x", pady=(0,20))

        ttk.Label(ctrl, text="Step-by-step Algorithm:").pack(anchor="w")
        self.algo_var = tk.StringVar(value="BFS")
        algos = ["BFS", "DFS", "Dijkstra", "Prim", "Kruskal", "Floyd–Warshall"]
        ttk.OptionMenu(ctrl, self.algo_var, algos[0], *algos).pack(fill="x", pady=(0,10))
        ttk.Button(ctrl, text="Start Animation", command=self._start_animation)\
            .pack(fill="x", pady=(0,5))
        self.next_step_btn = ttk.Button(ctrl, text="Next Step", command=self._next_step, state="disabled")
        self.next_step_btn.pack(fill="x")

        self.info = ttk.Label(ctrl, text="", wraplength=220, justify="left")
        self.info.pack(pady=(20,0), anchor="w")

        # ─── Drawing Area ────────────────────────────────────────────────────
        self.figure = plt.Figure(figsize=(7,7))
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Key bindings
        self.bind("<KeyPress-Right>", self._on_right_press)
        self.bind("<KeyRelease-Right>", self._on_right_release)

    # ─── Helpers ────────────────────────────────────────────────────────────
    def _interrupt_and(self, fn):
        self._stop_animation()
        fn()

    def _stop_animation(self):
        self.animating = False
        self.steps.clear(); self.step_index = 0
        self.current_path.clear(); self.current_edges.clear()
        self.next_step_btn.config(state="disabled")
        if self._auto_after_id:
            self.after_cancel(self._auto_after_id)
            self._auto_after_id = None
        self._holding = False

    def _on_nodes_change(self, v):
        n = int(float(v))
        self.start_scale.config(to=n-1)
        self.end_scale.config(to=n-1)
        if self.start_scale.get() > n-1: self.start_scale.set(n-1)
        if self.end_scale.get() > n-1: self.end_scale.set(n-1)

    def _on_slider_change(self, _):
        # dynamic path highlight only after Floyd–Warshall completed
        if self.fw_done and not self.animating:
            self._highlight_fw_path()

    def _build_undirected(self, n):
        t = self.graph_type.get()
        if t=="Tree":    return nx.random_tree(n)
        if t=="Sparse":  return nx.gnm_random_graph(n, max(2*n,n-1))
        if t=="Dense":   return nx.gnm_random_graph(n,int(0.8*(n*(n-1)//2)))
        if t=="Complete":return nx.complete_graph(n)
        if t=="Grid":
            r=int(n**0.5) or 1; c=int(np.ceil(n/r))
            G=nx.grid_2d_graph(r,c)
            return nx.convert_node_labels_to_integers(G)
        return nx.Graph()

    def _orient(self, G):
        D=nx.DiGraph(); D.add_nodes_from(G.nodes(data=True))
        for u,v in G.edges():
            if random.random()<0.5: D.add_edge(u,v)
            else: D.add_edge(v,u)
        return D

    def _assign_weights(self, G):
        for u,v in G.edges():
            G[u][v]['weight']=random.randint(1,20)

    def _draw_graph(self, highlight_nodes=None, highlight_edges=None):
        self.ax.clear()
        pos=nx.spring_layout(self.G,seed=42)
        nx.draw(self.G,pos,ax=self.ax,
                with_labels=True,node_size=300,node_color='lightblue',
                edge_color='gray',arrows=self.is_directed.get())
        if self.is_weighted.get():
            labels=nx.get_edge_attributes(self.G,'weight')
            nx.draw_networkx_edge_labels(self.G,pos,edge_labels=labels,ax=self.ax)
        if highlight_edges:
            nx.draw_networkx_edges(self.G,pos,edgelist=highlight_edges,
                                   width=3,edge_color='orange',
                                   arrows=self.is_directed.get(),ax=self.ax)
        if highlight_nodes:
            nx.draw_networkx_nodes(self.G,pos,nodelist=highlight_nodes,
                                   node_size=400,node_color='orange',ax=self.ax)
        self.canvas.draw()

    # ─── Generation & Plotting ─────────────────────────────────────────────
    def _on_generate(self):
        self._stop_animation()
        self.fw_done=False
        n=self.node_scale.get()
        self.G0=self._build_undirected(n)
        self.G=self.is_directed.get() and self._orient(self.G0) or self.G0.copy()
        if self.is_weighted.get(): self._assign_weights(self.G)
        self._draw_graph()
        self.info.config(text="Graph generated.")

    def _on_plot_bfs_dfs(self):
        self._stop_animation()
        max_n=self.node_scale.get()
        sizes=np.unique(np.linspace(2,max_n,500,dtype=int))
        bfs_t,dfs_t=[],[]
        for n in sizes:
            G0=self._build_undirected(n)
            G=self.is_directed.get() and self._orient(G0) or G0
            t0=time.perf_counter(); list(nx.bfs_edges(G,0)); bfs_t.append(time.perf_counter()-t0)
            t0=time.perf_counter(); list(nx.dfs_edges(G,0)); dfs_t.append(time.perf_counter()-t0)
        self.ax.clear()
        self.ax.plot(sizes,bfs_t,label='BFS',marker='.',linewidth=1)
        self.ax.plot(sizes,dfs_t,label='DFS',marker='.',linewidth=1)
        self.ax.set_xlabel('Nodes');self.ax.set_ylabel('Time (s)')
        self.ax.set_title('BFS vs DFS');self.ax.legend();self.canvas.draw()

    def _on_plot_sp(self):
        self._stop_animation()
        self.fw_done=False
        max_n=self.node_scale.get()
        s_ui,e_ui=self.start_scale.get(),self.end_scale.get()
        sizes=np.unique(np.linspace(2,max_n,100,dtype=int))
        dij_t,fw_t=[],[]
        for n in sizes:
            s,e=min(s_ui,n-1),min(e_ui,n-1)
            G0=self._build_undirected(n)
            G=self.is_directed.get() and self._orient(G0) or G0
            if self.is_weighted.get(): self._assign_weights(G)
            else:
                for u,v in G.edges(): G[u][v]['weight']=1
            t0=time.perf_counter(); nx.dijkstra_path(G,s,e,weight='weight'); dij_t.append(time.perf_counter()-t0)
            t0=time.perf_counter(); dist=nx.floyd_warshall(G,weight='weight'); fw_t.append(time.perf_counter()-t0)
        self.ax.clear()
        self.ax.plot(sizes,dij_t,label='Dijkstra',marker='.',linewidth=1)
        self.ax.plot(sizes,fw_t,label='Floyd–Warshall',marker='.',linewidth=1)
        self.ax.set_xlabel('Nodes');self.ax.set_ylabel('Time (s)')
        self.ax.set_title('Dijkstra vs Floyd–Warshall');self.ax.legend();self.canvas.draw()

    def _on_plot_mst(self):
        self._stop_animation()
        max_n=self.node_scale.get()
        sizes=np.unique(np.linspace(2,max_n,100,dtype=int))
        prim_t,kruskal_t=[],[]
        for n in sizes:
            G0=self._build_undirected(n)
            if self.is_weighted.get(): self._assign_weights(G0)
            else:
                for u,v in G0.edges(): G0[u][v]['weight']=1
            t0=time.perf_counter(); nx.minimum_spanning_tree(G0,algorithm='prim',weight='weight'); prim_t.append(time.perf_counter()-t0)
            t0=time.perf_counter(); nx.minimum_spanning_tree(G0,algorithm='kruskal',weight='weight'); kruskal_t.append(time.perf_counter()-t0)
        self.ax.clear()
        self.ax.plot(sizes,prim_t,label="Prim's",marker='.',linewidth=1)
        self.ax.plot(sizes,kruskal_t,label="Kruskal's",marker='.',linewidth=1)
        self.ax.set_xlabel('Nodes');self.ax.set_ylabel('Time (s)')
        self.ax.set_title("Prim vs Kruskal");self.ax.legend();self.canvas.draw()

    # ─── Recording Steps ───────────────────────────────────────────────────
    def _record_bfs(self,s,t):
        visited,parent,q={s},{},[s]
        self.steps=[([],[],f"Start BFS from {s}")]
        while q:
            u=q.pop(0)
            self.steps.append(([u],[],f"Visit {u}"))
            if u==t: break
            for v in self.G[u]:
                if v not in visited:
                    visited.add(v); parent[v]=u; q.append(v)
                    self.steps.append(([],[(u,v)],f"Queue {u}→{v}"))
        path,cur=[],t
        while cur in parent:
            path.insert(0,cur); cur=parent[cur]
        if path: path.insert(0,s)
        self.current_path,self.current_edges=path,[]

    def _record_dfs(self,s,t):
        visited,parent,stack=set(),{},[s]
        self.steps=[([],[],f"Start DFS from {s}")]
        while stack:
            u=stack.pop()
            if u not in visited:
                visited.add(u); self.steps.append(([u],[],f"Visit {u}"))
                if u==t: break
                for v in self.G[u]:
                    if v not in visited:
                        parent[v]=u; stack.append(v)
                        self.steps.append(([],[(u,v)],f"Push {u}→{v}"))
        path,cur=[],t
        while cur in parent:
            path.insert(0,cur); cur=parent[cur]
        if path: path.insert(0,s)
        self.current_path,self.current_edges=path,[]

    def _record_dijkstra(self,s,t):
        dist,prev={n:float('inf') for n in self.G},{}
        dist[s]=0; Q=set(self.G)
        self.steps=[([],[],f"Start Dijkstra {s}→{t}")]
        while Q:
            u=min(Q,key=lambda x:dist[x]); Q.remove(u)
            self.steps.append(([u],[],f"Settle {u}"))
            for v in self.G[u]:
                w=self.G[u][v]['weight']
                if dist[u]+w<dist[v]:
                    dist[v]=dist[u]+w; prev[v]=u
                    self.steps.append(([],[(u,v)],f"Relax {u}→{v}"))
        path,cur=[],t
        while cur in prev:
            path.insert(0,cur); cur=prev[cur]
        if path: path.insert(0,s)
        self.current_path,self.current_edges=path,[]

    def _record_prim(self):
        self.steps=[([],[], "Start Prim")]
        visited={next(iter(self.G))}
        edges=[(self.G[u][v]['weight'],u,v) for u in visited for v in self.G[u]]
        heapq.heapify(edges); mst=[]
        while edges and len(visited)<self.G.number_of_nodes():
            w,u,v=heapq.heappop(edges)
            if v in visited: continue
            visited.add(v); mst.append((u,v))
            self.steps.append(([],[(u,v)],f"Add {u}–{v}"))
            for x in self.G[v]:
                if x not in visited:
                    heapq.heappush(edges,(self.G[v][x]['weight'],v,x))
        self.current_path,self.current_edges=[],mst

    def _record_kruskal(self):
        self.steps=[([],[],"Start Kruskal")]
        edges=sorted((self.G[u][v]['weight'],u,v) for u,v in self.G.edges())
        uf={n:n for n in self.G}; mst=[]
        def find(u):
            while uf[u]!=u: u=uf[u]
            return u
        def union(a,b): uf[find(a)]=find(b)
        for w,u,v in edges:
            if find(u)!=find(v):
                union(u,v); mst.append((u,v))
                self.steps.append(([],[(u,v)],f"Add {u}–{v}"))
        self.current_path,self.current_edges=[],mst

    def _record_floyd_warshall(self):
        n=self.G.number_of_nodes()
        dist={i:{j:(self.G[i][j]['weight'] if self.G.has_edge(i,j) else float('inf')) for j in self.G} for i in self.G}
        next_hop={i:{j:(j if self.G.has_edge(i,j) else None) for j in self.G} for i in self.G}
        for i in self.G: dist[i][i]=0; next_hop[i][i]=i
        self.steps=[([],[],"Init Floyd–Warshall")]
        for k in self.G:
            for i in self.G:
                for j in self.G:
                    if dist[i][k]+dist[k][j]<dist[i][j]:
                        old=dist[i][j]
                        dist[i][j]=dist[i][k]+dist[k][j]
                        next_hop[i][j]=next_hop[i][k]
                        self.steps.append(([],[(i,j)],f"Update dist[{i},{j}]{old}→{dist[i][j]}"))
        self.fw_next=next_hop
        self.fw_done=True
        self.current_path=[]; self.current_edges=[]

    # ─── Animation Control ─────────────────────────────────────────────────
    def _start_animation(self):
        if not hasattr(self,'G'):
            self.info.config(text="Generate a graph first."); return
        self._stop_animation()
        self.fw_done=False
        algo=self.algo_var.get()
        s,e=self.start_scale.get(),self.end_scale.get()
        if algo=="BFS":         self._record_bfs(s,e)
        elif algo=="DFS":       self._record_dfs(s,e)
        elif algo=="Dijkstra":
            if not self.is_weighted.get(): self.info.config(text="Dijkstra needs weights."); return
            self._record_dijkstra(s,e)
        elif algo=="Prim":
            if not self.is_weighted.get(): self.info.config(text="Prim needs weights."); return
            self._record_prim()
        elif algo=="Kruskal":
            if not self.is_weighted.get(): self.info.config(text="Kruskal needs weights."); return
            self._record_kruskal()
        elif algo=="Floyd–Warshall":
            if not self.is_weighted.get(): self.info.config(text="FW needs weights."); return
            self._record_floyd_warshall()
        self.step_index=0; self.animating=True
        self.next_step_btn.config(state="normal")
        self.info.config(text="Animation started. Use → or Next Step.")

    def _next_step(self):
        if not self.animating: return
        if self.step_index<len(self.steps):
            nodes,edges,desc=self.steps[self.step_index]
            self._draw_graph(highlight_nodes=nodes,highlight_edges=edges)
            self.info.config(text=desc)
            self.step_index+=1
        else:
            if self.current_path:
                edges=list(zip(self.current_path,self.current_path[1:]))
                self._draw_graph(highlight_nodes=self.current_path,highlight_edges=edges)
                self.info.config(text="Done: path highlighted.")
            elif self.current_edges:
                self._draw_graph(highlight_edges=self.current_edges)
                self.info.config(text="Done: MST highlighted.")
            else:
                self.info.config(text="Done.")
            self.animating=False
            self.next_step_btn.config(state="disabled")

    # ─── Arrow key handling ────────────────────────────────────────────────
    def _on_right_press(self,event):
        if not self.animating: return
        if not self._holding:
            self._holding=True
            self._next_step()
            self._auto_after_id=self.after(200,self._auto_next)

    def _auto_next(self):
        if self._holding and self.animating:
            self._next_step()
            self._auto_after_id=self.after(100,self._auto_next)

    def _on_right_release(self,event):
        self._holding=False
        if self._auto_after_id:
            self.after_cancel(self._auto_after_id)
            self._auto_after_id=None

    def _highlight_fw_path(self):
        # reconstruct path from next_hop
        i,j=self.start_scale.get(),self.end_scale.get()
        if self.fw_next and self.fw_next[i][j] is not None:
            path=[i]
            while i!=j:
                i=self.fw_next[i][j]
                path.append(i)
            edges=list(zip(path,path[1:]))
            self._draw_graph(highlight_nodes=path,highlight_edges=edges)
            cost_text=f"cost={len(path)-1}"  # exact cost from dist omitted here
            self.info.config(text=f"FW path: {'→'.join(map(str,path))}")

if __name__=="__main__":
    app=GraphExplorerUI()
    app.mainloop()
