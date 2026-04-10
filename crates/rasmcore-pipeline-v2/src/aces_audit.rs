//! ACES compliance audit — validates pipeline graphs for ACES correctness.
//!
//! Walks all nodes in a graph and reports any that are not ACES-compliant.
//! In strict mode, execution is blocked if violations are found.

use crate::graph::Graph;
use crate::node::AcesCompliance;

/// A single ACES compliance violation found in the graph.
#[derive(Debug, Clone)]
pub struct AcesViolation {
    /// Node ID of the violating node.
    pub node_id: u32,
    /// Compliance level reported by the node.
    pub compliance: AcesCompliance,
    /// Human-readable reason for the violation.
    pub reason: String,
}

impl std::fmt::Display for AcesViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "node {}: {:?} — {}",
            self.node_id, self.compliance, self.reason
        )
    }
}

/// Result of an ACES compliance audit.
#[derive(Debug, Clone)]
pub struct AcesAuditResult {
    /// Total nodes audited.
    pub total_nodes: u32,
    /// Nodes that are fully compliant.
    pub compliant: u32,
    /// Nodes that work in Log (also ACES-safe).
    pub log_compatible: u32,
    /// Nodes that are non-compliant.
    pub non_compliant: u32,
    /// Nodes with unknown compliance.
    pub unknown: u32,
    /// Detailed violations (non-compliant + unknown).
    pub violations: Vec<AcesViolation>,
}

impl AcesAuditResult {
    /// True if the entire graph is ACES-safe (no violations).
    pub fn is_aces_safe(&self) -> bool {
        self.violations.is_empty()
    }

    /// True if the graph is strictly compliant (no Unknown either).
    pub fn is_strictly_compliant(&self) -> bool {
        self.non_compliant == 0 && self.unknown == 0
    }
}

impl std::fmt::Display for AcesAuditResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "ACES Compliance Audit: {}/{} nodes safe",
            self.compliant + self.log_compatible,
            self.total_nodes
        )?;
        if self.violations.is_empty() {
            writeln!(f, "  PASS — all nodes are ACES-compliant")?;
        } else {
            writeln!(f, "  FAIL — {} violation(s):", self.violations.len())?;
            for v in &self.violations {
                writeln!(f, "    {v}")?;
            }
        }
        Ok(())
    }
}

impl Graph {
    /// Audit all nodes in the graph for ACES compliance.
    ///
    /// Returns a detailed report of compliance status per node.
    /// Non-compliant and Unknown nodes are reported as violations.
    pub fn validate_aces(&self) -> AcesAuditResult {
        let mut result = AcesAuditResult {
            total_nodes: self.node_count(),
            compliant: 0,
            log_compatible: 0,
            non_compliant: 0,
            unknown: 0,
            violations: Vec::new(),
        };

        for id in 0..self.node_count() {
            let node = &self.nodes[id as usize];
            let compliance = node.aces_compliance();

            match compliance {
                AcesCompliance::Compliant => result.compliant += 1,
                AcesCompliance::Log => result.log_compatible += 1,
                AcesCompliance::NonCompliant => {
                    result.non_compliant += 1;
                    result.violations.push(AcesViolation {
                        node_id: id,
                        compliance,
                        reason: "node declares itself as non-ACES-compliant".into(),
                    });
                }
                AcesCompliance::Unknown => {
                    result.unknown += 1;
                    result.violations.push(AcesViolation {
                        node_id: id,
                        compliance,
                        reason: "ACES compliance not audited — treat as unsafe".into(),
                    });
                }
            }
        }

        result
    }

    /// Set ACES strict mode. When enabled, `request_region` will validate
    /// ACES compliance before execution and return an error if violations
    /// are found.
    pub fn set_aces_strict(&mut self, strict: bool) {
        self.aces_strict = strict;
    }

    /// Check if ACES strict mode is enabled.
    pub fn aces_strict(&self) -> bool {
        self.aces_strict
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::color_space::ColorSpace;
    use crate::node::{AcesCompliance, Node, NodeInfo, PipelineError, Upstream};
    use crate::rect::Rect;

    struct CompliantNode;
    impl Node for CompliantNode {
        fn info(&self) -> NodeInfo {
            NodeInfo {
                width: 4,
                height: 4,
                color_space: ColorSpace::Linear,
            }
        }
        fn compute(&self, _r: Rect, _u: &mut dyn Upstream) -> Result<Vec<f32>, PipelineError> {
            Ok(vec![0.0; 64])
        }
        fn upstream_ids(&self) -> Vec<u32> {
            vec![]
        }
        fn aces_compliance(&self) -> AcesCompliance {
            AcesCompliance::Compliant
        }
    }

    struct LogNode;
    impl Node for LogNode {
        fn info(&self) -> NodeInfo {
            NodeInfo {
                width: 4,
                height: 4,
                color_space: ColorSpace::AcesCct,
            }
        }
        fn compute(&self, _r: Rect, _u: &mut dyn Upstream) -> Result<Vec<f32>, PipelineError> {
            Ok(vec![0.0; 64])
        }
        fn upstream_ids(&self) -> Vec<u32> {
            vec![]
        }
        fn aces_compliance(&self) -> AcesCompliance {
            AcesCompliance::Log
        }
    }

    struct NonCompliantNode;
    impl Node for NonCompliantNode {
        fn info(&self) -> NodeInfo {
            NodeInfo {
                width: 4,
                height: 4,
                color_space: ColorSpace::Srgb,
            }
        }
        fn compute(&self, _r: Rect, _u: &mut dyn Upstream) -> Result<Vec<f32>, PipelineError> {
            Ok(vec![0.0; 64])
        }
        fn upstream_ids(&self) -> Vec<u32> {
            vec![]
        }
        fn aces_compliance(&self) -> AcesCompliance {
            AcesCompliance::NonCompliant
        }
    }

    struct UnknownNode;
    impl Node for UnknownNode {
        fn info(&self) -> NodeInfo {
            NodeInfo {
                width: 4,
                height: 4,
                color_space: ColorSpace::Linear,
            }
        }
        fn compute(&self, _r: Rect, _u: &mut dyn Upstream) -> Result<Vec<f32>, PipelineError> {
            Ok(vec![0.0; 64])
        }
        fn upstream_ids(&self) -> Vec<u32> {
            vec![]
        }
        // default aces_compliance() → Unknown
    }

    #[test]
    fn all_compliant_graph_passes() {
        let mut g = Graph::new(0);
        g.add_node(Box::new(CompliantNode));
        g.add_node(Box::new(CompliantNode));
        g.add_node(Box::new(LogNode));

        let result = g.validate_aces();
        assert!(result.is_aces_safe());
        assert!(result.is_strictly_compliant());
        assert_eq!(result.compliant, 2);
        assert_eq!(result.log_compatible, 1);
        assert_eq!(result.violations.len(), 0);
    }

    #[test]
    fn non_compliant_node_reported() {
        let mut g = Graph::new(0);
        g.add_node(Box::new(CompliantNode));
        g.add_node(Box::new(NonCompliantNode));

        let result = g.validate_aces();
        assert!(!result.is_aces_safe());
        assert_eq!(result.non_compliant, 1);
        assert_eq!(result.violations.len(), 1);
        assert_eq!(result.violations[0].node_id, 1);
        assert_eq!(
            result.violations[0].compliance,
            AcesCompliance::NonCompliant
        );
    }

    #[test]
    fn unknown_node_is_violation() {
        let mut g = Graph::new(0);
        g.add_node(Box::new(CompliantNode));
        g.add_node(Box::new(UnknownNode));

        let result = g.validate_aces();
        assert!(!result.is_aces_safe());
        assert_eq!(result.unknown, 1);
        assert_eq!(result.violations.len(), 1);
    }

    #[test]
    fn mixed_graph_reports_all_violations() {
        let mut g = Graph::new(0);
        g.add_node(Box::new(CompliantNode));
        g.add_node(Box::new(NonCompliantNode));
        g.add_node(Box::new(UnknownNode));
        g.add_node(Box::new(LogNode));

        let result = g.validate_aces();
        assert!(!result.is_aces_safe());
        assert_eq!(result.total_nodes, 4);
        assert_eq!(result.compliant, 1);
        assert_eq!(result.log_compatible, 1);
        assert_eq!(result.non_compliant, 1);
        assert_eq!(result.unknown, 1);
        assert_eq!(result.violations.len(), 2);
    }

    #[test]
    fn aces_compliance_is_safe_check() {
        assert!(AcesCompliance::Compliant.is_aces_safe());
        assert!(AcesCompliance::Log.is_aces_safe());
        assert!(!AcesCompliance::NonCompliant.is_aces_safe());
        assert!(!AcesCompliance::Unknown.is_aces_safe());
    }

    #[test]
    fn audit_result_display() {
        let mut g = Graph::new(0);
        g.add_node(Box::new(CompliantNode));
        g.add_node(Box::new(NonCompliantNode));
        let result = g.validate_aces();
        let display = format!("{result}");
        assert!(display.contains("FAIL"));
        assert!(display.contains("1 violation"));
    }

    #[test]
    fn empty_graph_passes() {
        let g = Graph::new(0);
        let result = g.validate_aces();
        assert!(result.is_aces_safe());
        assert!(result.is_strictly_compliant());
        assert_eq!(result.total_nodes, 0);
    }
}
