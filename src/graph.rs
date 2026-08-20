pub struct Node {
    input_node_indices: Vec<usize>,
    output_node_indices: Vec<usize>,
    name: String,
}

pub struct Op {
    node_indices: Vec<usize>,
    name: String,
    op_type: String,
}

pub struct Graph {
    nodes: Vec<Node>,
    ops: Vec<Op>,
}
