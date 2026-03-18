//! Compute graph for optimization

use std::collections::HashMap;

use barq_core::error::{Error, Result};
use barq_core::tensor::Tensor;

/// Node in compute graph
#[derive(Debug, Clone)]
pub struct ComputeNode {
    /// Node ID
    pub id: usize,
    /// Operation type
    pub op: String,
    /// Input tensors
    pub inputs: Vec<String>,
    /// Output tensor
    pub output: Option<String>,
}

/// Compute graph
pub struct ComputeGraph {
    nodes: Vec<ComputeNode>,
    tensor_map: HashMap<String, Tensor>,
}

impl ComputeGraph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            tensor_map: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, op: String, inputs: Vec<String>) -> usize {
        let id = self.nodes.len();
        self.nodes.push(ComputeNode {
            id,
            op,
            inputs,
            output: None,
        });
        id
    }

    pub fn set_output(&mut self, node_id: usize, output: String) -> Result<()> {
        let node = self
            .nodes
            .get_mut(node_id)
            .ok_or_else(|| Error::tensor(format!("Invalid node ID: {}", node_id)))?;
        node.output = Some(output);
        Ok(())
    }

    pub fn add_tensor(&mut self, name: String, tensor: Tensor) {
        self.tensor_map.insert(name, tensor);
    }

    pub fn get_tensor(&self, name: &str) -> Option<&Tensor> {
        self.tensor_map.get(name)
    }

    pub fn execute(&mut self) -> Result<()> {
        // TODO: Implement graph execution
        Err(Error::Unsupported(
            "Graph execution not yet implemented".to_string(),
        ))
    }
}

impl Default for ComputeGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_graph() {
        let mut graph = ComputeGraph::new();
        let node_id = graph.add_node("add".to_string(), vec!["a".to_string(), "b".to_string()]);
        assert_eq!(node_id, 0);

        graph.set_output(node_id, "c".to_string()).unwrap();
    }
}
