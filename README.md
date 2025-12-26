src.model.transformer contains code for a decoder transformer model with energy-based capabilities.

# Debugging Modules
If you want to test a LightningModule directly in VScode (for example, test src.module.transformer.EnergyTransformer) you need to add a debug configuration .vscode/launch.json like this:

```json
{
    "name": "Python: Current File (Root Context)",
    "type": "debugpy",
    "request": "launch",
    "program": "${file}",
    "cwd": "${workspaceFolder}",
    "env": { "PYTHONPATH": "${workspaceFolder}" },
    "console": "integratedTerminal"
}
```