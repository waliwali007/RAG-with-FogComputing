create a network with zero tier "fog"
#to join the network : zerotier-cli join 632ea290858d13b4
nodes:

- name: Ahmed
  ip: "10.241.132.84"
  health_url: "http://10.241.132.84:5001/health"
- name: Nouha
  ip: "10.147.20.101"
  health_url: "http://10.241.140.160:5000/health"
- name: Hadeel
  ip: "10.241.178.195"
  health_url: ""

add IP Ahmed and Nouha to app.py nodes_url

each node should start the server :
Ahmed : python node_server.py --index pipeline/faiss_index_node1.index --metadata pipeline/chunks_metadata1.pkl --port 5001
Nouha : python node_server.py --index pipeline/faiss_index_node2.index --metadata pipeline/chunks_metadata2.pkl --port 5000

MASTER python -m pipeline.master_FOG
try the "Mode Distribué" si les nodes ne sont pas connectés il affiche "Les noeuds ne sont pas connectés."
sinon il affiche la réponse générée par le modèle
