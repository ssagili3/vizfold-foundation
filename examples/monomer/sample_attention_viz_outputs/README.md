# Sample attention visualization outputs

This directory is populated when you run:

```bash
python scripts/run_head_heatmap_network_demo.py
```

from the repository root. The script uses **synthetic** attention data (same file format as real OpenFold `--demo_attn` output) to generate:

- **head_heatmaps/** — One combined PNG with a grid of residue–residue heatmaps (one per attention head).
- **network_plots/** — Aggregated attention network (all heads combined, hub residues highlighted) and a per-head network grid.

To generate visualizations from **real** inference, run the full pipeline in `viz_attention_demo.ipynb` or `viz_attention_demo_base.ipynb`; the same heatmap and network code runs in the notebook and writes to `IMAGE_OUTPUT_DIR/head_heatmaps/` and `IMAGE_OUTPUT_DIR/network_plots/`.

See [Head_Visualization_Heatmap_Network.md](../../../docs/source/Head_Visualization_Heatmap_Network.md) for a full description and comparison with arc diagrams and 3D overlays.
